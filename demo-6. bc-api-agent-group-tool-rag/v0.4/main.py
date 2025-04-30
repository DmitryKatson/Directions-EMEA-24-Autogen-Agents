import asyncio
import os
import dotenv
from pathlib import Path
from typing import List, Dict, Any

# Core AgentChat components
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console

# Model and Qdrant components
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from qdrant_client import QdrantClient, models # Import models for search
from langchain_openai import AzureOpenAIEmbeddings

# Tool import
from bc_api_tool import call_business_central_api

# Load environment variables
# Assuming .env is one level up from v0.4 dir
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env') 
dotenv.load_dotenv(dotenv_path=dotenv_path, override=True)

async def main():

    # === Model Client Setup ===
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

    embedding_function = AzureOpenAIEmbeddings(
        azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL"), 
        openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    )

    # === Qdrant Client Setup ===
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    qdrant_collection_name = os.environ.get("QDRANT_COLLECTION", "bc_api_docs")
    
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables.")

    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True, 
        timeout=30
    )

    # === Qdrant Query Tool Definition ===
    async def query_qdrant_documentation(query: str) -> str:
        """Queries the Qdrant vector database for relevant Business Central API documentation based on the user query."""
        try:
            print(f"--- Qdrant Tool: Embedding query: '{query}' ---")
            query_vector = await embedding_function.aembed_query(query)
            
            print(f"--- Qdrant Tool: Searching collection '{qdrant_collection_name}' ---")
            search_result = qdrant_client.search(
                collection_name=qdrant_collection_name,
                query_vector=query_vector,
                limit=3,  # Retrieve top 3 results
                # score_threshold=0.5, # Optional: set a score threshold 
                with_payload=True # Ensure payload (like page_content) is returned
            )
            
            context = "\n\nRetrieved Documentation Sections:\n---\n"
            if not search_result:
                 context += "No relevant documentation found.\n"
            else:
                for i, hit in enumerate(search_result):
                    content = hit.payload.get('page_content', '[No content found]') if hit.payload else '[No payload]'
                    context += f"Section {i+1} (Score: {hit.score:.4f}):\n{content}\n---\n"
            
            print(f"--- Qdrant Tool: Returning context ---")
            return context
        except Exception as e:
            print(f"--- Qdrant Tool: Error querying Qdrant: {e} ---")
            return f"Error: Could not retrieve documentation from Qdrant. {e}"

    # === Agent Definitions ===
    
    # AssistantAgent (RAG via Tool + API Tool Capable)
    bc_api_assistant = AssistantAgent(
        name="BC_API_Assistant",
        description="Business Central API agent. Uses retrieved documentation and API tools to answer questions.",
        model_client=model_client,
        tools=[query_qdrant_documentation, call_business_central_api], # Include BOTH tools
        # memory=None, # Not using built-in memory
        reflect_on_tool_use=True,
        system_message="You are a helpful AI assistant specialized in Business Central API. "
                       "To answer the user's question: "
                       "1. First, use the `query_qdrant_documentation` tool with a relevant search query based on the user's request to retrieve API documentation context. "
                       "2. Then, analyze the retrieved documentation context AND use the `call_business_central_api` tool to construct and execute the appropriate API call. "
                       "3. Finally, analyze the results from the API call and provide a clear, final answer to the user. "
                       "If an API tool call fails, analyze the error based on the context and try again with a corrected approach. "
                       "End your final answer with 'TASK_COMPLETE' on a new line when the task is complete.",
    )

    # === Team Setup ===
    
    # Define termination conditions
    max_message_termination = MaxMessageTermination(10)
    text_termination = TextMentionTermination("TASK_COMPLETE") 
    termination = max_message_termination | text_termination

    # Create the RoundRobinGroupChat team
    group_chat = RoundRobinGroupChat(
        participants=[bc_api_assistant], # Only the assistant is needed
        termination_condition=termination,
    )

    # === Run Conversation ===
    
    user_message = "Which customers owes money to the company?"

    print(f"User: {user_message}")
    stream = group_chat.run_stream(task=user_message)
    await Console(stream, output_stats=True)

    # === Cleanup ===
    await model_client.close()
    # qdrant_client might not have an explicit async close, handled by context typically

if __name__ == "__main__":
    asyncio.run(main())