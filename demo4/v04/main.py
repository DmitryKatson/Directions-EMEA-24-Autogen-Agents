import asyncio
import os
import sys
import dotenv
from pathlib import Path
from typing import List, Dict, Any, Tuple
from functools import partial
import json # For formatting arguments/results nicely

# Core AgentChat components
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import BaseChatMessage, TextMessage

# Model and Qdrant components
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from qdrant_client import QdrantClient, models # Import models for search
from langchain_openai import AzureOpenAIEmbeddings

# Tool import
from bc_api_tool import call_business_central_api

dotenv.load_dotenv()

def create_agent_team():
    """
    Creates and initializes the model client, agent (with RAG and API tools), 
    and group chat for the Demo 4 setup.

    Returns:
        Tuple containing model_client, bc_api_assistant, and group_chat
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

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

    # Note: QdrantClient might be better managed if long-lived, but for simplicity 
    # we create it here. Consider passing it in if the app structure evolves.
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
                score_threshold=0.7, # Optional: set a score threshold 
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
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            return f"Error: Could not retrieve documentation from Qdrant. {e}"

    # === Agent Definitions ===
    
    # AssistantAgent (RAG via Tool + API Tool Capable)
    bc_api_assistant = AssistantAgent(
        name="BC_API_Assistant",
        description="Business Central API agent. Uses retrieved documentation and API tools to answer questions.",
        model_client=model_client,
        # Pass the locally defined query tool and the imported API tool
        tools=[query_qdrant_documentation, call_business_central_api], 
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

    # Return the necessary components
    return model_client, bc_api_assistant, group_chat

async def run_chat(group_chat, prompt: str) -> List[Dict[str, Any]]:
    """
    Run the agent team chat for a given prompt and collect detailed event logs.

    Args:
        group_chat: The RoundRobinGroupChat instance to use
        prompt: The user prompt to process

    Returns:
        List of dictionaries, where each dictionary represents a detailed event log.
    """
    print(f"--- Starting run_chat for prompt: \'{prompt}\' ---")
    detailed_logs: List[Dict[str, Any]] = []

    # Append the initial user prompt
    detailed_logs.append({"type": "user_prompt", "content": prompt})

    stream = group_chat.run_stream(task=prompt)

    async for event_data in stream:
        event_type_name = type(event_data).__name__
        log_entry = {"type": event_type_name, "data": None}
        source_name = getattr(getattr(event_data, 'source', None), 'name', str(getattr(event_data, 'source', 'unknown')))

        if event_type_name == "TextMessage" or isinstance(event_data, TextMessage):
            log_entry["data"] = {
                "source": source_name,
                "content": event_data.to_text()
            }
        elif event_type_name == "ToolCallRequestMessage":
             # Assume structure based on type name
             log_entry["data"] = {
                 "source": source_name,
                 "tool_calls": [
                     {"id": getattr(tc, 'id', None), 
                      "name": getattr(tc, 'name', None), 
                      "arguments": getattr(tc, 'arguments', None)}
                     for tc in getattr(event_data, 'tool_calls', [])
                 ]
             }
        elif event_type_name == "ToolCallResultMessage":
             # Assume structure based on type name
             log_entry["data"] = {
                 "source": source_name,
                 "tool_call_results": [
                     {"call_id": getattr(res, 'call_id', None), 
                      "name": getattr(res, 'name', None), 
                      "content": getattr(res, 'content', None), 
                      "is_error": getattr(res, 'is_error', False)}
                     for res in getattr(event_data, 'tool_call_results', [])
                 ]
             }
        else:
            # Catch-all for other types, try to represent them
            try:
                log_entry["data"] = event_data.model_dump() # If pydantic model
            except AttributeError:
                try:
                   log_entry["data"] = vars(event_data) # If standard object
                except TypeError:
                   log_entry["data"] = repr(event_data) # Fallback

        detailed_logs.append(log_entry)

    print(f"--- run_chat finished. Collected {len(detailed_logs)} detailed logs. ---")
    return detailed_logs

async def main():
    """Main function for running as a standalone script"""
    # Create agent and team components
    model_client, _, group_chat = create_agent_team()

    try:
        user_message = "Which customers owes money to the company?"

        print(f"\n--- Running Demo 4 (RAG + Tool) standalone for: '{user_message}' ---")
        stream = group_chat.run_stream(task=user_message)
        await Console(stream, output_stats=True)

    finally:
        # Ensure model client is closed
        print("\n--- Closing model client --- ")
        if model_client:
            await model_client.close()
        # QdrantClient doesn't typically require explicit async close in this usage pattern

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
         print(f"An error occurred during standalone execution: {e}")
         import traceback
         traceback.print_exc()
         if isinstance(e, NotImplementedError) and "bc_api_tool could not be imported" in str(e):
             print("Please ensure bc_api_tool.py is in the same directory and this script is run as part of a package (e.g., via the Streamlit app).")