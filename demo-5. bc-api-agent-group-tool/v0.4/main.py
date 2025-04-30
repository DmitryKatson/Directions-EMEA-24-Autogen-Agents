import asyncio
import os
import dotenv
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent # Removed UserProxyAgent import
from autogen_agentchat.teams import RoundRobinGroupChat # Use a v0.4 team
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination # Termination conditions
from autogen_agentchat.ui import Console # For streaming output
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

    # AssistantAgent: Calls the tool and answers
    bc_api_assistant = AssistantAgent(
        name="BC_API_Assistant",
        description="Business Central API agent. Responsible for constructing and executing the API call and answering.",
        model_client=model_client,
        tools=[call_business_central_api], # Tool passed directly
        reflect_on_tool_use=True, # Explain tool results
        system_message="You are a helpful AI assistant specialized in Business Central API. "
                       "Use the provided `call_business_central_api` tool to query Business Central data based on the user's request. "
                       "Analyze the tool results and provide a clear answer to the user. "
                       "If a tool call fails, analyze the error and try again with a corrected approach. "
                       "End your final answer with 'TASK_COMPLETE' on a new line when the task is complete.",
    )

    # Define termination conditions
    max_message_termination = MaxMessageTermination(10)
    text_termination = TextMentionTermination("TASK_COMPLETE") # Assistant signals completion
    termination = max_message_termination | text_termination

    # Create the RoundRobinGroupChat team
    group_chat = RoundRobinGroupChat(
        participants=[bc_api_assistant], # Only the assistant is needed
        termination_condition=termination,
    )

    # Define the user's task
    user_message = "Which customers owes money to the company?"

    # Start the conversation using the team and stream the output
    print(f"User: {user_message}")
    stream = group_chat.run_stream(task=user_message)
    await Console(stream, output_stats=True)

    # Close the model client connection
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())