import asyncio
import os
import sys
import importlib
import dotenv
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from typing import List, Tuple, Callable, Any
from autogen_agentchat.messages import BaseChatMessage

# --- Import bc_api_tool --- 
# Try relative import first (for package context like Streamlit)
# Fallback to direct import (for running script standalone)
try:
    from .bc_api_tool import call_business_central_api
except ImportError:
    print("Relative import failed. Trying direct import for bc_api_tool...")
    try:
        import bc_api_tool
        call_business_central_api = bc_api_tool.call_business_central_api
        print("Direct import successful.")
    except ImportError as e:
        print(f"Direct import failed: {e}")
        # Define a dummy function if all imports fail
        def call_business_central_api(*args, **kwargs):
            raise NotImplementedError(f"bc_api_tool could not be imported via relative or direct path. Error: {e}")

# Load environment variables
dotenv.load_dotenv()

def create_agent_team():
    """
    Creates and initializes the model client, agent, and group chat for the Demo 3 setup.

    Returns:
        Tuple containing model_client, bc_api_assistant, and group_chat
    """
    
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

    # Create the Business Central API agent with the tool (imported at module level)
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

    # Define termination conditions for the team
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


async def run_chat(group_chat, prompt: str) -> List[BaseChatMessage]:
    """
    Run the agent team chat for a given prompt and collect all messages.

    Args:
        group_chat: The RoundRobinGroupChat instance to use
        prompt: The user prompt to process

    Returns:
        List of BaseChatMessage objects from the conversation
    """
    print(f"--- Starting run_chat for prompt: '{prompt}' ---")
    all_messages: List[BaseChatMessage] = []

    stream = group_chat.run_stream(task=prompt)

    async for message in stream:
        if isinstance(message, BaseChatMessage):
            all_messages.append(message)

    print(f"--- run_chat finished. Collected {len(all_messages)} messages. ---")
    return all_messages


async def main():
    """Main function for running as a standalone script"""
    # Create agent and team components
    model_client, _, group_chat = create_agent_team() # Unpack needed components

    try:
        # Define the user's task for standalone execution
        user_message = "Which customers owes money to the company?"
        # user_message = "Get total sales for customer 10000" # Another example

        # Start the conversation using the team and stream the output to Console
        print(f"\n--- Running Demo 3 (Tool Use) standalone for: '{user_message}' ---")
        stream = group_chat.run_stream(task=user_message)
        await Console(stream, output_stats=True) # output_stats can be helpful

    finally:
        # Ensure model client is closed
        print("\n--- Closing model client ---")
        await model_client.close()

if __name__ == "__main__":
    # Updated error handling for standalone run
    try:
        asyncio.run(main())
    except Exception as e:
         # Catch any exception during standalone execution, including potential import errors
         print(f"An error occurred during standalone execution: {e}")
         # If the dummy tool was created due to import failure, the NotImplementedError will be caught here.
         if isinstance(e, NotImplementedError) and "bc_api_tool could not be imported" in str(e):
             print("Please ensure bc_api_tool.py is in the same directory as main.py.")