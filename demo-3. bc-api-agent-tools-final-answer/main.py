import autogen
import os
import dotenv
# Import the call_business_central_api function
from bc_api_tool import call_business_central_api


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

    user_proxy = autogen.UserProxyAgent(
        name="User_Agent",
        code_execution_config=False,
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=4,
    )

    # Create the Business Central API agent
    bc_api_assistant = autogen.AssistantAgent(
        name="BC_API_Assistant",
        description="Business Central API agent. Responsible to construct the API call",
        llm_config=llm_config,
        code_execution_config=False,
        human_input_mode="NEVER",
        system_message="You are a helpful AI assistant specialized in Business Central API. "
                       "You can help with querying Business Central data using the provided API tool. "
                       "Return 'TERMINATE' when the task is done.",        
        max_consecutive_auto_reply=4,
    )

    # Register the Business Central API function to the two agents
    autogen.register_function(
        call_business_central_api,
        caller=bc_api_assistant,  # The assistant agent can suggest calls to the BC API
        executor=user_proxy,  # The user proxy agent can execute the BC API calls
        name="call_business_central_api",
        description="A function to call Business Central API endpoints with various parameters",
    )

    # Create the GroupChat object
    groupchat = autogen.GroupChat(
        agents=[user_proxy, bc_api_assistant],
        messages=[],
        max_round=10,
        allowed_or_disallowed_speaker_transitions={
            user_proxy: [bc_api_assistant],
        },
        speaker_transitions_type="allowed",
    )

    # Create the GroupChatManager
    manager = autogen.GroupChatManager(
        groupchat=groupchat, llm_config=llm_config
    )

    # Start the conversation
    groupchat_result = user_proxy.initiate_chat(
        manager,
        message="Which customers owes money to the company?"
    )

if __name__ == "__main__":
    main()