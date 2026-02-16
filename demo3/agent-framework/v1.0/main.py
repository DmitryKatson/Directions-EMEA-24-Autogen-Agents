import asyncio
import os
import dotenv
from typing import List, Dict, Any, Tuple

from agent_framework.azure import AzureOpenAIChatClient
from bc_api_tool import call_business_central_api
import bc_api_tool

# Load environment variables
streamlit_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'streamlit', '.env')
root_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), '.env')

if os.path.exists(streamlit_env_path):
    dotenv.load_dotenv(streamlit_env_path)
elif os.path.exists(root_env_path):
    dotenv.load_dotenv(root_env_path)
else:
    dotenv.load_dotenv()


def create_agent_team():
    """
    Create and initialize the Agent Framework client and agent for Demo 3.
    
    Returns:
        Tuple containing client and agent
    """
    # Create the Azure OpenAI Chat client
    client = AzureOpenAIChatClient(
        endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
        deployment_name=os.environ.get("AZURE_OPENAI_API_MODEL"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    )
    
    # Create the agent with the call_business_central_api tool
    agent = client.as_agent(
        name="BC_API_Assistant",
        instructions=(
            "You are a helpful AI assistant specialized in Business Central API. "
            "You can help with querying Business Central data using the provided call_business_central_api tool. "
            "Execute the tool call based on the user's request, analyze the results, and provide a clear answer to the user. "
            "If a tool call fails, analyze the error carefully. "
            "When a field name is rejected, fetch a small sample (top=5) without the select parameter to discover the actual property names, then retry with the correct names. "
            "Common endpoints: customers, items, salesOrders, salesInvoices, purchaseOrders, generalLedgerEntries, accounts, vendors. "
            "Always provide the final answer in clear, natural language."
        ),
        tools=[call_business_central_api],
        description="Business Central API agent responsible for constructing and executing API calls and answering user questions.",
    )
    
    return client, agent


async def run_chat(agent, prompt: str) -> List[Dict[str, Any]]:
    """
    Run the agent for a given prompt and collect detailed event logs.
    
    Args:
        agent: The Agent instance to use
        prompt: The user prompt to process
        
    Returns:
        List of dictionaries representing detailed event logs
    """
    print(f"--- Starting run_chat for prompt: '{prompt}' ---")
    detailed_logs: List[Dict[str, Any]] = []
    
    # Clear tool call logs from previous runs
    bc_api_tool._tool_call_logs = []
    
    # Append the initial user prompt
    detailed_logs.append({"type": "user_prompt", "content": prompt})
    
    # Track current message being accumulated from streaming tokens
    current_message = None
    current_source = "BC_API_Assistant"
    last_tool_log_index = 0
    
    try:
        # Run the agent with streaming
        async for event in agent.run(prompt, stream=True):
            event_type = type(event).__name__
            
            # Check if there are new tool logs to inject
            if last_tool_log_index < len(bc_api_tool._tool_call_logs):
                # Save current accumulated message before inserting tool logs
                if current_message is not None:
                    detailed_logs.append(current_message)
                    current_message = None
                
                # Insert new tool logs
                detailed_logs.extend(bc_api_tool._tool_call_logs[last_tool_log_index:])
                last_tool_log_index = len(bc_api_tool._tool_call_logs)
            
            # Map Agent Framework events to the expected log format
            if hasattr(event, 'text') and event.text:
                # Accumulate streaming tokens into complete messages
                if current_message is not None:
                    current_message["data"]["content"] += event.text
                else:
                    # Start new message
                    current_message = {
                        "type": "TextMessage",
                        "data": {
                            "source": current_source,
                            "content": event.text
                        }
                    }
            
            # Check for tool call events
            if hasattr(event, 'tool_calls') and event.tool_calls:
                # Save current accumulated message first
                if current_message is not None:
                    detailed_logs.append(current_message)
                    current_message = None
                
                # Tool call request
                tool_calls = []
                for tc in event.tool_calls:
                    tool_calls.append({
                        "id": getattr(tc, 'id', None),
                        "name": getattr(tc, 'name', None),
                        "arguments": getattr(tc, 'arguments', None)
                    })
                
                log_entry = {
                    "type": "ToolCallRequestMessage",
                    "data": {
                        "source": current_source,
                        "tool_calls": tool_calls
                    }
                }
                detailed_logs.append(log_entry)
            
            # Check for tool results
            if hasattr(event, 'tool_results') and event.tool_results:
                # Save current accumulated message first
                if current_message is not None:
                    detailed_logs.append(current_message)
                    current_message = None
                
                # Tool call results
                tool_results = []
                for res in event.tool_results:
                    tool_results.append({
                        "call_id": getattr(res, 'call_id', None),
                        "name": getattr(res, 'name', None),
                        "content": getattr(res, 'content', str(res)),
                        "is_error": getattr(res, 'is_error', False)
                    })
                
                log_entry = {
                    "type": "ToolCallResultMessage",
                    "data": {
                        "source": current_source,
                        "tool_call_results": tool_results
                    }
                }
                detailed_logs.append(log_entry)
        
        # Don't forget to append the last accumulated message
        if current_message is not None:
            detailed_logs.append(current_message)
        
        # Append any remaining tool call logs
        if last_tool_log_index < len(bc_api_tool._tool_call_logs):
            detailed_logs.extend(bc_api_tool._tool_call_logs[last_tool_log_index:])
        
        print(f"--- run_chat finished. Collected {len(detailed_logs)} detailed logs. ---")
        return detailed_logs
        
    except Exception as e:
        print(f"--- run_chat error: {e} ---")
        # Append any accumulated message before error
        if current_message is not None:
            detailed_logs.append(current_message)
        # Add error message to logs
        detailed_logs.append({
            "type": "TextMessage",
            "data": {
                "source": "system",
                "content": f"Error during execution: {str(e)}"
            }
        })
        return detailed_logs


async def main():
    """Main function for running as a standalone script"""
    client, agent = create_agent_team()
    
    try:
        # Demo message
        message = "Which customers owes money to the company?"
        
        print(f"\n--- Running Demo 3 (Agent Framework) standalone for: '{message}' ---")
        
        # Run with the agent
        result = await agent.run(message)
        print("\nAgent Response:")
        print(result.text)
        
    finally:
        print("\n--- Demo complete ---")


if __name__ == "__main__":
    asyncio.run(main())
