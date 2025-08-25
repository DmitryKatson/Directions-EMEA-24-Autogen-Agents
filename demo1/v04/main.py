import asyncio
import os
import dotenv
from typing import List, Dict, Any, Tuple
import json
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core import CancellationToken

# Load environment variables when the module is imported
# Try to load from streamlit directory first, then from root
streamlit_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'streamlit', '.env')
root_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')

if os.path.exists(streamlit_env_path):
    dotenv.load_dotenv(streamlit_env_path)
elif os.path.exists(root_env_path):
    dotenv.load_dotenv(root_env_path)
else:
    dotenv.load_dotenv()  # Fallback to default behavior

def create_agent_team():
    """
    Create and initialize the model client, agents, and group chat for the BC API demo.
    
    Returns:
        Tuple containing model_client, bc_api_agent, executor_agent, and group_chat
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the model client for Azure OpenAI
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

    # Construct the path to the system message file
    system_message_path = os.path.join(script_dir, 'bc_api_agent_system_message.txt')

    # Read the system message from the text file
    with open(system_message_path, 'r') as file:
        bc_api_agent_system_message = file.read()

    # Create the Business Central API agent
    bc_api_agent = AssistantAgent(
        name="bc_api_agent",
        description="Business Central API agent. Responsible to choose and configure the API call",
        system_message=bc_api_agent_system_message,
        model_client=model_client,
    )

    # Create the code executor agent
    code_executor_agent = CodeExecutorAgent(
        name="executor_agent",  # Renamed for clarity from user_proxy_agent
        code_executor=LocalCommandLineCodeExecutor(work_dir=os.path.join(script_dir, 'coding')),
    )

    # Set up combined termination conditions
    max_message_termination = MaxMessageTermination(8)
    text_termination = TextMentionTermination("TERMINATE")
    
    # Use OR operator (|) to stop if either condition is met
    termination = max_message_termination | text_termination

    # Set up the group chat
    group_chat = RoundRobinGroupChat(
        [bc_api_agent, code_executor_agent], 
        termination_condition=termination
    )
    
    return model_client, bc_api_agent, code_executor_agent, group_chat

async def run_chat(group_chat, prompt: str) -> List[Dict[str, Any]]:
    """
    Run the agent team chat for a given prompt and collect detailed event logs.
    
    Args:
        group_chat: The RoundRobinGroupChat instance to use
        prompt: The user prompt to process
        
    Returns:
        List of dictionaries, where each dictionary represents a detailed event log.
    """
    print(f"--- Starting run_chat for prompt: \'{prompt}\' ---")
    detailed_logs: List[Dict[str, Any]] = []
    
    # Append the initial user prompt
    detailed_logs.append({"type": "user_prompt", "content": prompt})
    
    stream = group_chat.run_stream(task=prompt)
    
    async for event_data in stream:
        event_type_name = type(event_data).__name__
        log_entry = {"type": event_type_name, "data": None}
        # Safely get source name
        source_name = getattr(getattr(event_data, 'source', None), 'name', str(getattr(event_data, 'source', 'unknown')))

        if event_type_name == "TextMessage" or isinstance(event_data, TextMessage):
            # Special handling for CodeExecutorAgent messages which might contain code blocks
            content = event_data.to_text()
            log_entry["data"] = {
                "source": source_name,
                "content": content
            }
        # Demo 1 doesn't use tools, so no need for ToolCallRequest/Result handling
        # But we can add a placeholder if needed later
        # elif event_type_name == "ToolCallRequestMessage": ... 
        # elif event_type_name == "ToolCallResultMessage": ...
        else:
            # Catch-all for other types (like CodeExecution*) 
            try:
                # Try to get code execution details if available
                if event_type_name.startswith("CodeExecution") and hasattr(event_data, 'execution_result'):
                     log_entry["data"] = {
                         "source": source_name,
                         "execution_result": getattr(event_data, 'execution_result')
                     }
                elif hasattr(event_data, 'model_dump'): # If pydantic model
                    log_entry["data"] = event_data.model_dump() 
                else:
                   log_entry["data"] = vars(event_data) # If standard object
            except (AttributeError, TypeError):
                   log_entry["data"] = repr(event_data) # Fallback

        detailed_logs.append(log_entry)
            
    print(f"--- run_chat finished. Collected {len(detailed_logs)} detailed logs. ---")
    return detailed_logs

async def main():
    """Main function for running as a standalone script"""
    # Create agents and team
    model_client, bc_api_agent, executor_agent, group_chat = create_agent_team()

    try:
        # Demo 1: Show me cash accounts
        message = "Show me cash accounts"

        # Demo 2: List me the Profit & Loss CoA
        # message = "List me the Profit & Loss CoA"

        # Start the conversation with streaming output
        stream = group_chat.run_stream(task=message)
        await Console(stream, output_stats=True)
    finally:
        # Ensure model client is closed even if an exception occurs
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())








