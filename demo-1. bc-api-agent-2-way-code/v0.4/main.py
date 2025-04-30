import asyncio
import os
import dotenv
from typing import List, Tuple
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import BaseChatMessage
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core import CancellationToken

# Load environment variables when the module is imported
dotenv.load_dotenv()

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
            # No need to print here since this would duplicate console output
    
    print(f"--- run_chat finished. Collected {len(all_messages)} messages. ---")
    return all_messages

async def main():
    """Main function for running as a standalone script"""
    # Create agents and team
    model_client, bc_api_agent, executor_agent, group_chat = create_agent_team()

    try:
        # Demo 1: List me the Cash CoA
        message = "List me the Cash CoA"

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








