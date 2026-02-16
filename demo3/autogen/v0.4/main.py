import asyncio
import os
import sys
import dotenv
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json # Import json

# Core AgentChat components
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
# Import only the base message types
from autogen_agentchat.messages import BaseChatMessage, TextMessage 

# Model components
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from bc_api_tool import call_business_central_api

# Load environment variables
# Try to load from streamlit directory first, then from root
# Note: We're now 4 levels deep (demo3/autogen/v0.4/main.py), so we need 4 dirname() calls
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
streamlit_env_path = os.path.join(workspace_root, 'streamlit', '.env')
root_env_path = os.path.join(workspace_root, '.env')

if os.path.exists(streamlit_env_path):
    dotenv.load_dotenv(streamlit_env_path)
elif os.path.exists(root_env_path):
    dotenv.load_dotenv(root_env_path)
else:
    dotenv.load_dotenv()  # Fallback to default behavior

def create_agent_team():
    """
    Creates and initializes the model client, agent, termination conditions, 
    and group chat for the Demo 3 setup.

    Returns:
        Tuple containing model_client, bc_api_assistant, and group_chat
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # === Model Client Setup ===
    model_client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.environ.get("AZURE_OPENAI_API_MODEL"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
        model=os.environ.get("AZURE_OPENAI_API_MODEL"), 
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        temperature=0,
        max_tokens=4000,
        timeout=120,
    )

    # === Agent Definition ===
    bc_api_assistant = AssistantAgent(
        name="BC_API_Assistant",
        description="Business Central API agent responsible for constructing and executing API calls and answering user questions.",
        model_client=model_client,
        tools=[call_business_central_api], 
        reflect_on_tool_use=True, # Allows agent to comment on tool results
        system_message="You are a helpful AI assistant specialized in Business Central API. "
                       "You can help with querying Business Central data using the provided `call_business_central_api` tool. "
                       "Execute the tool call based on the user's request, analyze the results, and provide a clear answer to the user. "
                       "If a tool call fails, analyze the error and try again with a corrected approach. "
                       "End your final answer with 'TASK_COMPLETE' on a new line when the task is complete.",
    )

    # === Team Setup ===
    
    # Define termination conditions
    max_message_termination = MaxMessageTermination(10)
    text_termination = TextMentionTermination("TASK_COMPLETE") 
    termination = max_message_termination | text_termination

    # Create a single-agent team (RoundRobinGroupChat with one agent)
    # Even with one agent, GroupChat provides the structure for termination, context management etc.
    group_chat = RoundRobinGroupChat(
        participants=[bc_api_assistant],
        termination_condition=termination,
    )

    return model_client, bc_api_assistant, group_chat

async def run_chat(group_chat, prompt: str, on_event=None) -> List[Dict[str, Any]]: # Updated return type
    """
    Run the agent team chat for a given prompt and collect detailed event logs.

    Args:
        group_chat: The RoundRobinGroupChat instance to use
        prompt: The user prompt to process
        on_event: Optional callback function for real-time event streaming

    Returns:
        List of dictionaries, where each dictionary represents a detailed event log.
    """
    print(f"--- Starting run_chat for prompt: \'{prompt}\' ---")
    detailed_logs: List[Dict[str, Any]] = []

    # Append the initial user prompt
    detailed_logs.append({"type": "user_prompt", "content": prompt})

    stream = group_chat.run_stream(task=prompt)

    # Track current message being accumulated from streaming tokens
    current_message = None
    current_source = None

    async for event_data in stream:
        event_type_name = type(event_data).__name__
        # Safely get source name
        source_name = getattr(getattr(event_data, 'source', None), 'name', str(getattr(event_data, 'source', 'unknown')))

        if event_type_name == "TextMessage" or isinstance(event_data, TextMessage):
            # Accumulate streaming tokens into complete messages
            content = event_data.to_text()
            
            # Emit streaming token event for real-time display
            if on_event:
                on_event({"type": "streaming_token", "source": source_name, "content": content})
            
            # If this is from the same source, accumulate the content
            if current_source == source_name and current_message is not None:
                current_message["data"]["content"] += content
            else:
                # Save previous message if it exists
                if current_message is not None:
                    detailed_logs.append(current_message)
                
                # Start new message
                current_message = {
                    "type": "TextMessage",
                    "data": {
                        "source": source_name,
                        "content": content
                    }
                }
                current_source = source_name
        elif event_type_name == "ToolCallRequestMessage":
            # For non-text messages, save current accumulated message first
            if current_message is not None:
                detailed_logs.append(current_message)
                current_message = None
                current_source = None
            
            log_entry = {"type": event_type_name, "data": None}
            log_entry["data"] = {
                 "source": source_name,
                 "tool_calls": [
                     {"id": getattr(tc, 'id', None), 
                      "name": getattr(tc, 'name', None), 
                      "arguments": getattr(tc, 'arguments', None)}
                     for tc in getattr(event_data, 'tool_calls', [])
                 ]
             }
            detailed_logs.append(log_entry)
            
            # Emit tool call events for real-time display
            if on_event:
                for tc in getattr(event_data, 'tool_calls', []):
                    import json
                    try:
                        args_dict = json.loads(getattr(tc, 'arguments', '{}'))
                    except (json.JSONDecodeError, TypeError):
                        args_dict = {}
                    on_event({
                        "type": "tool_call",
                        "source": source_name,
                        "tool_name": getattr(tc, 'name', 'unknown'),
                        "arguments": args_dict
                    })
        elif event_type_name == "ToolCallResultMessage":
            # For non-text messages, save current accumulated message first
            if current_message is not None:
                detailed_logs.append(current_message)
                current_message = None
                current_source = None
            
            log_entry = {"type": event_type_name, "data": None}
            log_entry["data"] = {
                 "source": source_name,
                 "tool_call_results": [
                     {"call_id": getattr(res, 'call_id', None), 
                      "name": getattr(res, 'name', None), 
                      "content": getattr(res, 'content', None), 
                      "is_error": getattr(res, 'is_error', False)}
                     for res in getattr(event_data, 'tool_call_results', [])
                 ]
             }
            detailed_logs.append(log_entry)
            
            # Emit tool result events for real-time display
            if on_event:
                for res in getattr(event_data, 'tool_call_results', []):
                    on_event({
                        "type": "tool_result",
                        "source": source_name,
                        "tool_name": getattr(res, 'name', 'unknown'),
                        "content": getattr(res, 'content', ''),
                        "is_error": getattr(res, 'is_error', False)
                    })
        else:
            # For non-text messages, save current accumulated message first
            if current_message is not None:
                detailed_logs.append(current_message)
                current_message = None
                current_source = None
            
            # Catch-all for other types
            log_entry = {"type": event_type_name, "data": None}
            try:
                log_entry["data"] = event_data.model_dump() # If pydantic model
            except AttributeError:
                try:
                   log_entry["data"] = vars(event_data) # If standard object
                except TypeError:
                   log_entry["data"] = repr(event_data) # Fallback

            detailed_logs.append(log_entry)

    # Don't forget to append the last accumulated message
    if current_message is not None:
        detailed_logs.append(current_message)

    print(f"--- run_chat finished. Collected {len(detailed_logs)} detailed logs. ---")
    return detailed_logs

async def main():
    """Main function for running as a standalone script"""
    # Create agent and team components
    model_client, _, group_chat = create_agent_team()

    try:
        user_message = "Which customers owes money to the company?"

        print(f"\n--- Running Demo 3 (Agent + Tool) standalone for: '{user_message}' ---")
        stream = group_chat.run_stream(task=user_message)
        await Console(stream, output_stats=True)

    finally:
        # Ensure model client is closed
        print("\n--- Closing model client --- ")
        if model_client: # Check if model_client was successfully created
            await model_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
         print(f"An error occurred during standalone execution: {e}")
         import traceback
         traceback.print_exc()
         if isinstance(e, NotImplementedError) and "bc_api_tool could not be imported" in str(e):
             print("Please ensure bc_api_tool.py is in the same directory and this script is run as part of a package (e.g., via the Streamlit app).")