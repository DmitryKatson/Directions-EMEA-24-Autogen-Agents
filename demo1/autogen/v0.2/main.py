import autogen
import os
import dotenv

dotenv.load_dotenv()

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the LLM config
    config_list = [
        {
            "model": os.environ.get("AZURE_OPENAI_API_MODEL"),  # Specify your Azure OpenAI deployment name
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),  # API Key from environment variable
            "base_url": os.environ.get("AZURE_OPENAI_API_BASE"),  # Base URL for Azure OpenAI
            "api_type": "azure",  # Specify the API type as Azure
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),  # API version
            "max_tokens": 4000,  # Maximum tokens for the response
        }
    ]

    llm_config = {
        "config_list": config_list,
        "temperature": 0,  # Controls randomness in responses
        "timeout": 120,  # Timeout for requests
    }

    # Construct the path to the system message file
    system_message_path = os.path.join(script_dir, 'bc_api_agent_system_message.txt')

    # Read the system message from the text file
    with open(system_message_path, 'r') as file:
        bc_api_agent_system_message = file.read()

    # Set the max turns for the conversation
    max_turns = 4

    # Create the Business Central API agent
    bc_api_agent = autogen.AssistantAgent(
        name="bc_api_agent",
        description="Business Central API agent. Responsible to choose and configure the API call",
        llm_config=llm_config,
        code_execution_config=False,
        human_input_mode="NEVER",
        system_message=bc_api_agent_system_message,
        max_consecutive_auto_reply=max_turns,
    )

    # Create the user proxy agent
    user_proxy_agent = autogen.UserProxyAgent(
        name="user_proxy_agent",
        llm_config=False,
        code_execution_config={"work_dir": os.path.join(script_dir, 'coding'), "use_docker": False},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=max_turns,
    )

    # Demo 1: Show me cash accounts
    message = "Show me cash accounts"

    # Demo 2: List me the Profit & Loss CoA
    # message = "List me the Profit & Loss CoA"

    # Start the conversation
    user_proxy_agent.initiate_chat(bc_api_agent, message=message)

if __name__ == "__main__":
    main()