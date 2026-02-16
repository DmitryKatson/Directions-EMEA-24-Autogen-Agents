import asyncio
import os
import sys
import json
import dotenv
import subprocess
import tempfile
import hashlib
from typing import List, Dict, Any, Tuple
from pathlib import Path

from agent_framework import tool
from agent_framework.azure import AzureOpenAIChatClient

# Load environment variables
streamlit_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'streamlit', '.env')
root_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), '.env')

if os.path.exists(streamlit_env_path):
    dotenv.load_dotenv(streamlit_env_path)
elif os.path.exists(root_env_path):
    dotenv.load_dotenv(root_env_path)
else:
    dotenv.load_dotenv()

# Global variable to store tool call logs for the current run
_tool_call_logs = []

# Custom tool for executing Python code locally
@tool(approval_mode="never_require")
def execute_python_code(code: str) -> str:
    """
    Execute Python code locally and return the output.
    
    Args:
        code: The Python code to execute as a string
        
    Returns:
        The stdout and stderr output from executing the code
    """
    global _tool_call_logs
    
    # Log the tool call request
    tool_call_id = f"call_{hashlib.sha256(code.encode()).hexdigest()[:8]}"
    _tool_call_logs.append({
        "type": "ToolCallRequestMessage",
        "data": {
            "source": "bc_api_agent",
            "tool_calls": [{
                "id": tool_call_id,
                "name": "execute_python_code",
                "arguments": json.dumps({"code": code})
            }]
        }
    })
    
    try:
        # Create a temporary directory for code execution
        script_dir = os.path.dirname(os.path.abspath(__file__))
        work_dir = os.path.join(script_dir, 'coding')
        os.makedirs(work_dir, exist_ok=True)
        
        # Create a unique filename for this code execution
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        filename = f"tmp_code_{code_hash}.py"
        filepath = os.path.join(work_dir, filename)
        
        # Write code to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Execute the code using subprocess
        result = subprocess.run(
            [sys.executable, filepath],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=work_dir
        )
        
        # Combine stdout and stderr
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        
        if result.returncode != 0:
            output += f"\n\nExecution failed with exit code {result.returncode}"
        
        final_output = output if output else "Code executed successfully with no output."
        
        # Log the tool call result
        _tool_call_logs.append({
            "type": "ToolCallResultMessage",
            "data": {
                "source": "bc_api_agent",
                "tool_call_results": [{
                    "call_id": tool_call_id,
                    "name": "execute_python_code",
                    "content": final_output,
                    "is_error": result.returncode != 0 if 'result' in locals() else False
                }]
            }
        })
        
        return final_output
        
    except subprocess.TimeoutExpired:
        error_msg = "Error: Code execution timed out after 120 seconds."
        # Log the tool call result with error
        _tool_call_logs.append({
            "type": "ToolCallResultMessage",
            "data": {
                "source": "bc_api_agent",
                "tool_call_results": [{
                    "call_id": tool_call_id,
                    "name": "execute_python_code",
                    "content": error_msg,
                    "is_error": True
                }]
            }
        })
        return error_msg
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        # Log the tool call result with error
        _tool_call_logs.append({
            "type": "ToolCallResultMessage",
            "data": {
                "source": "bc_api_agent",
                "tool_call_results": [{
                    "call_id": tool_call_id,
                    "name": "execute_python_code",
                    "content": error_msg,
                    "is_error": True
                }]
            }
        })
        return error_msg


def create_agent_team():
    """
    Create and initialize the Agent Framework client and agent for Demo 1.
    
    Returns:
        Tuple containing client and agent
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read the system message
    system_message_path = os.path.join(script_dir, 'bc_api_agent_system_message.txt')
    with open(system_message_path, 'r') as file:
        system_message = file.read()
    
    # Create the Azure OpenAI Chat client
    client = AzureOpenAIChatClient(
        endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
        deployment_name=os.environ.get("AZURE_OPENAI_API_MODEL"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    )
    
    # Create the agent with the execute_python_code tool
    agent = client.as_agent(
        name="bc_api_agent",
        instructions=system_message,
        tools=[execute_python_code],
    )
    
    return client, agent


async def run_chat(agent, prompt: str, on_event=None) -> List[Dict[str, Any]]:
    """
    Run the agent for a given prompt and collect detailed event logs.
    
    Args:
        agent: The Agent instance to use
        prompt: The user prompt to process
        on_event: Optional callback function for real-time event streaming
        
    Returns:
        List of dictionaries representing detailed event logs
    """
    global _tool_call_logs
    print(f"--- Starting run_chat for prompt: '{prompt}' ---")
    detailed_logs: List[Dict[str, Any]] = []
    
    # Clear tool call logs from previous runs
    _tool_call_logs = []
    
    # Append the initial user prompt
    detailed_logs.append({"type": "user_prompt", "content": prompt})
    
    # Track current message being accumulated from streaming tokens
    current_message = None
    current_source = "bc_api_agent"
    last_tool_log_index = 0
    
    try:
        # Run the agent with streaming
        async for event in agent.run(prompt, stream=True):
            event_type = type(event).__name__
            
            # Check if there are new tool logs to inject
            if last_tool_log_index < len(_tool_call_logs):
                # Save current accumulated message before inserting tool logs
                if current_message is not None:
                    detailed_logs.append(current_message)
                    current_message = None
                
                # Emit tool call events for real-time display
                if on_event:
                    for tl in _tool_call_logs[last_tool_log_index:]:
                        if tl["type"] == "ToolCallRequestMessage":
                            for tc in tl["data"]["tool_calls"]:
                                import json
                                try:
                                    args_dict = json.loads(tc["arguments"])
                                except (json.JSONDecodeError, TypeError):
                                    args_dict = {}
                                on_event({
                                    "type": "tool_call",
                                    "source": tl["data"]["source"],
                                    "tool_name": tc["name"],
                                    "arguments": args_dict
                                })
                        elif tl["type"] == "ToolCallResultMessage":
                            for res in tl["data"]["tool_call_results"]:
                                on_event({
                                    "type": "tool_result",
                                    "source": tl["data"]["source"],
                                    "tool_name": res["name"],
                                    "content": res["content"],
                                    "is_error": res["is_error"]
                                })
                
                # Insert new tool logs
                detailed_logs.extend(_tool_call_logs[last_tool_log_index:])
                last_tool_log_index = len(_tool_call_logs)
            
            # Map Agent Framework events to the expected log format
            if hasattr(event, 'text') and event.text:
                # Emit streaming token event for real-time display
                if on_event:
                    on_event({"type": "streaming_token", "source": current_source, "content": event.text})
                
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
        
        # Don't forget to append the last accumulated message
        if current_message is not None:
            detailed_logs.append(current_message)
        
        # Append any remaining tool call logs
        if last_tool_log_index < len(_tool_call_logs):
            detailed_logs.extend(_tool_call_logs[last_tool_log_index:])
        
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
        message = "Show me cash accounts"
        
        print(f"\n--- Running Demo 1 (Agent Framework) standalone for: '{message}' ---")
        
        # Run with the agent
        result = await agent.run(message)
        print("\nAgent Response:")
        print(result.text)
        
    finally:
        print("\n--- Demo complete ---")


if __name__ == "__main__":
    asyncio.run(main())
