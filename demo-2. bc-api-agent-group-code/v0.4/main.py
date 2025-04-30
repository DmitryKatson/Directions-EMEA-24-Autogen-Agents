import asyncio
import os
import dotenv
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from typing import Sequence, List, Tuple
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage

# Load environment variables
dotenv.load_dotenv()

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


async def run_chat(group_chat, prompt: str) -> List[BaseChatMessage]:
    """
    Run the agent team chat for a given prompt and collect all messages.

    Args:
        group_chat: The SelectorGroupChat instance to use
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








