import asyncio
import os
import sys
import dotenv
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Core AgentChat components
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import BaseChatMessage

# Model components
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from bc_api_tool import call_business_central_api

dotenv.load_dotenv()

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