import autogen
from autogen import initiate_chats
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
    
    user_proxy = autogen.UserProxyAgent(
        name="User_Agent",
        code_execution_config=False,
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
    )

    # Create the Business Central API agent
    bc_api_coder = autogen.AssistantAgent(
        name="Coder_Agent",
        description="Business Central API agent. Responsible to choose and write the code to call the API",
        llm_config=llm_config,
        code_execution_config=False,
        human_input_mode="NEVER",
        system_message=bc_api_agent_system_message,
        max_consecutive_auto_reply=2,
    )

    # Create the executor agent
    executor = autogen.UserProxyAgent(
        name="Executor_Agent",
        system_message="Executor. Execute the code written by the Coder and report the result.",
        human_input_mode="NEVER",
        llm_config=False,
        code_execution_config={"work_dir": os.path.join(script_dir, 'coding'), "use_docker": False},
        max_consecutive_auto_reply=2,
    )

    # Create the answer composer agent
    answer_composer_agent = autogen.AssistantAgent(
        name="Answer_Composer_Agent",
        description="Answer composer agent. Responsible to compose the answer from the API call",
        llm_config=llm_config,
        code_execution_config=False,
        human_input_mode="NEVER",
        system_message="You are a helpful assistant that composes the answer from the API call. Be concise and accurate.",
        max_consecutive_auto_reply=2,
    )

    groupchat = autogen.GroupChat(
        agents=[user_proxy, bc_api_coder, executor, answer_composer_agent],
        messages=[],
        max_round=10,
        allowed_or_disallowed_speaker_transitions={
            user_proxy: [bc_api_coder, answer_composer_agent],
            bc_api_coder: [executor],
            executor: [bc_api_coder, answer_composer_agent]
        },
        speaker_transitions_type="allowed",
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat, llm_config=llm_config
        )

    groupchat_result = user_proxy.initiate_chat(
        manager,
    )

if __name__ == "__main__":
    main()








