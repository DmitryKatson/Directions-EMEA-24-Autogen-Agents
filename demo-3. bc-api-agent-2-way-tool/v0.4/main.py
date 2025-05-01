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

# Import the custom tool
# Ensure bc_api_tool.py is in the same directory or accessible via PYTHONPATH
# Use an absolute approach to import the tool that works in both standalone and Streamlit contexts
_bc_api_tool = None

def _import_bc_api_tool():
    """
    Dynamically import the bc_api_tool module, searching in the current directory
    and handling import errors with a more robust approach.
    """
    global _bc_api_tool
    
    # If already imported successfully, return it
    if _bc_api_tool is not None:
        return _bc_api_tool.call_business_central_api

    # Try different import approaches
    try:
        # First approach: direct import (works when running as standalone)
        import bc_api_tool as tool_module
        _bc_api_tool = tool_module
        return _bc_api_tool.call_business_central_api
    except ImportError:
        try:
            # Second approach: import from current file's directory
            file_dir = os.path.dirname(os.path.abspath(__file__))
            if file_dir not in sys.path:
                sys.path.insert(0, file_dir)
                
            # Try import after path modification
            import bc_api_tool as tool_module
            _bc_api_tool = tool_module
            return _bc_api_tool.call_business_central_api
        except ImportError as e:
            # Third approach: try to load module directly from file
            try:
                tool_path = os.path.join(file_dir, "bc_api_tool.py")
                if os.path.exists(tool_path):
                    spec = importlib.util.spec_from_file_location("bc_api_tool", tool_path)
                    tool_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(tool_module)
                    _bc_api_tool = tool_module
                    return _bc_api_tool.call_business_central_api
                else:
                    print(f"bc_api_tool.py not found at {tool_path}")
                    raise FileNotFoundError(f"bc_api_tool.py not found at {tool_path}")
            except Exception as e2:
                # If all attempts fail, create a dummy function with helpful error message
                print(f"All import approaches failed: {e2}")
                def dummy_tool(*args, **kwargs):
                    raise NotImplementedError("bc_api_tool could not be imported. "
                                            "Error: " + str(e) + 
                                            ". Make sure the file exists in the same directory.")
                return dummy_tool

# Load environment variables
dotenv.load_dotenv()

def create_agent_team():
    """
    Creates and initializes the model client, agent, and group chat for the Demo 3 setup.

    Returns:
        Tuple containing model_client, bc_api_assistant, and group_chat
    """
    # Try to get the tool function
    call_business_central_api = _import_bc_api_tool()
    
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
        tools=[call_business_central_api], # Assign the imported tool
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
    # Added robust error handling for the import during standalone run
    try:
        asyncio.run(main())
    except ImportError as e:
        print(f"Import error: {e}")
        if "bc_api_tool" in str(e):
            print("Could not run main: bc_api_tool.py failed to import.")
            print("Make sure bc_api_tool.py is in the same directory as this script.")
    except Exception as e:
         print(f"An error occurred during main execution: {e}")