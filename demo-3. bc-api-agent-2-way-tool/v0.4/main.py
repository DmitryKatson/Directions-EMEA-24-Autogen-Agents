import asyncio
import os
import dotenv
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
# Import the call_business_central_api function
from bc_api_tool import call_business_central_api

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

    # Create the Business Central API agent with the tool
    bc_api_assistant = AssistantAgent(
        name="BC_API_Assistant",
        description="Business Central API agent responsible for constructing and executing API calls and answering user questions.",
        model_client=model_client,
        tools=[call_business_central_api],
        reflect_on_tool_use=True,
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
    group_chat = RoundRobinGroupChat(
        participants=[bc_api_assistant],
        termination_condition=termination,
    )

    # Define the user's task
    user_message = "Which customers owes money to the company?"
    
    # Start the conversation using the team and stream the output
    print(f"User: {user_message}")
    stream = group_chat.run_stream(task=user_message)
    await Console(stream)

    # Close the model client connection
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())