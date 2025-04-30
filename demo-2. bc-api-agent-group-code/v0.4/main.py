import asyncio
import os
import dotenv
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core import CancellationToken
from typing import Sequence
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage

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
    bc_api_coder = AssistantAgent(
        name="Coder_Agent",
        description="Business Central API agent. Responsible to choose and write the code to call the API",
        system_message=bc_api_agent_system_message,
        model_client=model_client,
    )

    # Create the executor agent
    executor = CodeExecutorAgent(
        name="Executor_Agent",
        description="Executor. Execute the code written by the Coder and report the result.",
        code_executor=LocalCommandLineCodeExecutor(work_dir=os.path.join(script_dir, 'coding')),
    )

    # Create the answer composer agent
    answer_composer_agent = AssistantAgent(
        name="Answer_Composer_Agent",
        description="Answer composer agent. Responsible to compose the the final answer to the user's question",
        system_message="You are a helpful assistant that composes the the final answer to the user's question. Be concise and accurate. End your final answer with 'TASK_COMPLETE' on a new line when the task is complete.",
        model_client=model_client,
    )
    
    # Define the termination conditions
    max_message_termination = MaxMessageTermination(20)  # Allow more messages for complex conversations
    text_termination = TextMentionTermination("TASK_COMPLETE")
    termination = max_message_termination | text_termination

    # Define the selector function to control conversation flow
    def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if len(messages) == 1:  # Initial message: user -> coder
            return "Coder_Agent"
        
        last_message = messages[-1]
        
        if last_message.source == "Coder_Agent":  # Coder -> Executor
            return "Executor_Agent"
        elif last_message.source == "Executor_Agent":  # Executor -> Answer composer or back to Coder if there was an error
            # Simple heuristic: if the executor's message contains "error" or "exception", go back to the coder
            content = last_message.to_text().lower()
            if "error" in content or "exception" in content:
                return "Coder_Agent"
            else:
                return "Answer_Composer_Agent"
        elif last_message.source == "Answer_Composer_Agent":
            # End the conversation after the answer composer responds
            return None
        
        return None  # Default: end the conversation

    # Set up the SelectorGroupChat
    group_chat = SelectorGroupChat(
        participants=[bc_api_coder, executor, answer_composer_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_func=selector_func,
    )

    # Start the conversation with streaming output
    user_message = "List me all unpaid sales invoices for the last year"
    stream = group_chat.run_stream(task=user_message)
    await Console(stream, output_stats=True)
    
    # Close the connection to the model client
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())








