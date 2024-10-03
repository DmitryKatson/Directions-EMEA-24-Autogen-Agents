import autogen
import os
import dotenv
from bc_api_tool import call_business_central_api
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from chromadb.utils import embedding_functions
from qdrant_client import QdrantClient

dotenv.load_dotenv()

def main():
    config_list = [
        {
            "model": os.environ.get("AZURE_OPENAI_API_MODEL"),
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
            "base_url": os.environ.get("AZURE_OPENAI_API_BASE"),
            "api_type": "azure",
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
            "max_tokens": 4000,
        }
    ]

    llm_config = {
        "config_list": config_list,
        "temperature": 0,
        "timeout": 120,
    }

    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY")
    )

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                    model_name=os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL"),
                    api_base=os.environ.get("AZURE_OPENAI_API_BASE"),
                    api_type="azure",
                    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
                )

    user_proxy = autogen.UserProxyAgent(
        name="User_Agent",
        code_execution_config=False,
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=4,
    )

    bc_api_assistant = autogen.AssistantAgent(
        name="BC_API_Assistant",
        description="Business Central API agent. Responsible to construct the API call",
        llm_config=llm_config,
        code_execution_config=False,
        human_input_mode="NEVER",
        system_message="You are a helpful AI assistant specialized in Business Central API. "
                       "You can help with querying Business Central data using the provided API tool. "
                       "Use the retrieved API documentation to construct accurate API calls. "
                       "Return 'TERMINATE' when the task is done.",        
        max_consecutive_auto_reply=4,
    )

    rag_agent = RetrieveUserProxyAgent(
            name="RAG_Agent",
            human_input_mode="NEVER",
            default_auto_reply="Reply `TERMINATE` if the task is done.",
            max_consecutive_auto_reply=4,
            retrieve_config={
                "task": "qa",
                "vector_db": "qdrant",
                "db_config": {
                    "client": qdrant_client,
                    "content_payload_key": "page_content",
                    "metadata_payload_key": "metadata",
                },
                "collection_name": "bc_api_docs",
                "embedding_function": openai_ef,
                "distance_threshold": 0.77,
                "n_results": 5,
                "docs_path": None,
            },
            code_execution_config=False,
            description="Assistant who has extra content retrieval power from Qdrant to retrieve Microsoft Dynamics 365 Business Central API documentation"
        )

    autogen.register_function(
        call_business_central_api,
        caller=bc_api_assistant,
        executor=user_proxy,
        name="call_business_central_api",
        description="A function to call Business Central API endpoints with various parameters",
    )

    groupchat = autogen.GroupChat(
        agents=[user_proxy, rag_agent, bc_api_assistant],
        messages=[],
        max_round=10,
        # allowed_or_disallowed_speaker_transitions={
        #     user_proxy: [bc_api_assistant],
        #     bc_api_assistant: [rag_agent, user_proxy],
        # },
        # speaker_transitions_type="allowed",
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat, llm_config=llm_config
    )

    # user_proxy.initiate_chat(
    #     manager,
    #     message="Which customers owes money to the company?"
    # )

    rag_agent.initiate_chat(
        manager,
        message=rag_agent.message_generator,
        problem="Which customers owes money to the company?"
    )


if __name__ == "__main__":
    main()