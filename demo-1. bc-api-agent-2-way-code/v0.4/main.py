import asyncio
import os
import dotenv
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core import CancellationToken
import dotenv

dotenv.load_dotenv()

async def main():
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
        name="user_proxy_agent", 
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

    # Demo 1: List me the Cash CoA
    message = "List me the Cash CoA"

    # Demo 2: List me the Profit & Loss CoA
    # message = "List me the Profit & Loss CoA"

    # Start the conversation with streaming output
    stream = group_chat.run_stream(task=message)
    await Console(stream, output_stats=True)
    
    # Close the connection to the model client
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())








