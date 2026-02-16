import asyncio
import os
import dotenv
from typing import Sequence, List, Tuple, Dict, Any
import json
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage

# Load environment variables
# Try to load from streamlit directory first, then from root
# Note: We're now 4 levels deep (demo2/autogen/v0.4/main.py), so we need 4 dirname() calls
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
streamlit_env_path = os.path.join(workspace_root, 'streamlit', '.env')
root_env_path = os.path.join(workspace_root, '.env')

if os.path.exists(streamlit_env_path):
    dotenv.load_dotenv(streamlit_env_path)
elif os.path.exists(root_env_path):
    dotenv.load_dotenv(root_env_path)
else:
    dotenv.load_dotenv()  # Fallback to default behavior

# Define the custom selector prompt
SELECTOR_PROMPT = """You are the conductor managing a team of AI agents.
Read the following conversation history and agent descriptions.
Select the agent that should speak next according to the specified workflow.

Workflow:
1. The user initiates the request.
2. Coder_Agent receives the request, writes the Python code for the API call, and explains it.
3. Executor_Agent receives the code from Coder_Agent, executes it, and reports the full result (output or error trace).
4. Coder_Agent receives the execution result from Executor_Agent.
    a. If there was an execution error, Coder_Agent must debug, provide corrected code, and explain the fix. Then, go back to step 3 (Executor_Agent).
    b. If the execution was successful, Coder_Agent must analyze the output (e.g., identify data structure, confirm required info is present). If more steps are needed (like a follow-up API call), Coder_Agent writes the next code block and goes back to step 3 (Executor_Agent).
    c. If execution was successful AND the Coder_Agent determines all necessary information is gathered and processed, Coder_Agent should explicitly state this and summarize the key findings briefly.
5. Answer_Composer_Agent receives the confirmation and summary from Coder_Agent (from step 4c), composes the final user-friendly answer, and ends with TASK_COMPLETE.

Available Agents:
{roles}

Conversation History:
{history}

Based on the workflow and the history, select the next agent from {participants} to speak. Only output the name of the single agent selected.
"""

def create_agent_team():
    """
    Creates and initializes the model client, agents, and group chat for the Demo 2 BC API setup,
    using model-based speaker selection guided by a custom selector_prompt.

    Returns:
        Tuple containing model_client, bc_api_coder, executor, answer_composer_agent, and group_chat
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

    # Create the agents (descriptions are still helpful context for the selector prompt)
    bc_api_coder = AssistantAgent(
        name="Coder_Agent",
        description="Writes and refines Python code for Business Central API calls based on user requests and execution feedback.",
        system_message=bc_api_agent_system_message,
        model_client=model_client,
    )

    executor = CodeExecutorAgent(
        name="Executor_Agent",
        description="Executes Python code provided by Coder_Agent and reports results.",
        code_executor=LocalCommandLineCodeExecutor(work_dir=os.path.join(script_dir, 'coding')),
    )

    answer_composer_agent = AssistantAgent(
        name="Answer_Composer_Agent",
        description="Composes the final answer for the user after Coder_Agent confirms task completion and provides necessary results.",
        system_message="You compose the final user answer based *only* on the final processed information or confirmation provided by Coder_Agent. Wait for Coder_Agent's signal before composing the answer. End your response with TASK_COMPLETE.",
        model_client=model_client,
    )

    # Define the termination conditions
    max_message_termination = MaxMessageTermination(20)
    text_termination = TextMentionTermination("TASK_COMPLETE")
    termination = max_message_termination | text_termination

    # Set up the SelectorGroupChat WITH the custom selector_prompt
    group_chat = SelectorGroupChat(
        participants=[bc_api_coder, executor, answer_composer_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=SELECTOR_PROMPT, # Use the custom prompt
        allow_repeated_speaker=False # Keep default
    )

    return model_client, bc_api_coder, executor, answer_composer_agent, group_chat


async def run_chat(group_chat, prompt: str, on_event=None) -> List[Dict[str, Any]]:
    """
    Run the agent team chat for a given prompt and collect detailed event logs.

    Args:
        group_chat: The SelectorGroupChat instance to use
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
        else:
            # For non-text messages, save current accumulated message first
            if current_message is not None:
                detailed_logs.append(current_message)
                current_message = None
                current_source = None
            
            # Catch-all for other types (like CodeExecution*, AgentSelection*) 
            log_entry = {"type": event_type_name, "data": None}
            try:
                # Log AgentSelection events specifically if needed
                if event_type_name == "AgentSelection" and hasattr(event_data, 'selected_agent'):
                     selected_agent_name = getattr(getattr(event_data, 'selected_agent', None), 'name', 'unknown')
                     log_entry["data"] = {
                         "source": source_name, # Usually the SelectorGroupChat
                         "selected_agent": selected_agent_name
                     }
                     # Emit agent change event for real-time display
                     if on_event:
                         on_event({"type": "agent_change", "agent_name": selected_agent_name})
                # Log CodeExecution events
                elif event_type_name.startswith("CodeExecution") and hasattr(event_data, 'execution_result'):
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

    # Don't forget to append the last accumulated message
    if current_message is not None:
        detailed_logs.append(current_message)

    print(f"--- run_chat finished. Collected {len(detailed_logs)} detailed logs. ---")
    return detailed_logs


async def main():
    """Main function for running as a standalone script"""
    # Create agents and team
    model_client, _, _, _, group_chat = create_agent_team() # Unpack needed components

    try:
        # Example user message for standalone execution
        user_message = "List me all unpaid sales invoices for the last year"

        # Start the conversation with streaming output to console
        print(f"\n--- Running Demo 2 (Selector Prompt) standalone for: '{user_message}' ---")
        stream = group_chat.run_stream(task=user_message)
        await Console(stream, output_stats=True)

    finally:
        # Ensure model client is closed
        print("\n--- Closing model client ---")
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())








