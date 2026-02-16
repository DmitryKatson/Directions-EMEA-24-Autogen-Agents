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
from agent_framework.orchestrations import GroupChatBuilder

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
            "source": "Coder_Agent",
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
                "source": "Coder_Agent",
                "tool_call_results": [{
                    "call_id": tool_call_id,
                    "name": "execute_python_code",
                    "content": final_output,
                    "is_error": result.returncode != 0
                }]
            }
        })
        
        return final_output
        
    except subprocess.TimeoutExpired:
        error_msg = "Error: Code execution timed out after 120 seconds."
        _tool_call_logs.append({
            "type": "ToolCallResultMessage",
            "data": {
                "source": "Coder_Agent",
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
        _tool_call_logs.append({
            "type": "ToolCallResultMessage",
            "data": {
                "source": "Coder_Agent",
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
    Create and initialize the Agent Framework client, agents, and workflow for Demo 2.
    
    Returns:
        Tuple containing client and workflow
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read the system message for the Coder agent
    system_message_path = os.path.join(script_dir, 'bc_api_agent_system_message.txt')
    with open(system_message_path, 'r') as file:
        coder_system_message = file.read()
    
    # Create the Azure OpenAI Chat client
    client = AzureOpenAIChatClient(
        endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
        deployment_name=os.environ.get("AZURE_OPENAI_API_MODEL"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    )
    
    # Create the Coder agent with the execute_python_code tool
    coder_agent = client.as_agent(
        name="Coder_Agent",
        instructions=coder_system_message,
        description="Writes and executes Python code for Business Central API calls based on user requests. Reports results and debugs errors.",
        tools=[execute_python_code],
    )
    
    # Create the Answer Composer agent
    answer_composer_agent = client.as_agent(
        name="Answer_Composer_Agent",
        instructions="""You compose the final user-friendly answer based on the processed information provided by Coder_Agent. 
        
        Wait for Coder_Agent to explicitly state that the task is complete and provide a summary of findings.
        Then, take that information and format it into a clear, natural language response for the user.
        
        Your answer should:
        - Be conversational and easy to understand
        - Highlight the key findings
        - Present data in a well-formatted way
        - Answer the user's original question directly
        
        End your response with 'TASK_COMPLETE' on a new line when done.""",
        description="Composes the final user-friendly answer after Coder_Agent completes the task.",
    )
    
    # Create the orchestrator agent for managing the workflow
    orchestrator_agent = client.as_agent(
        name="orchestrator",
        instructions="""You are the conductor managing a team of AI agents.
        Read the conversation history and agent descriptions to select which agent should speak next.
        
        Workflow:
        1. User initiates the request
        2. Coder_Agent receives the request and writes/executes Python code for the API call
        3. Coder_Agent iterates on errors until successful execution
        4. When Coder_Agent reports "task complete" with a summary, select Answer_Composer_Agent
        5. Answer_Composer_Agent formats the final user-friendly answer and ends with TASK_COMPLETE
        
        Rules:
        - Start with Coder_Agent for the initial request
        - Keep selecting Coder_Agent while it's working on code/debugging
        - Only select Answer_Composer_Agent after Coder_Agent explicitly says the task is complete
        - If Answer_Composer_Agent has spoken, the conversation is done
        
        Select the appropriate agent name from the available agents.""",
    )
    
    # Build the group chat workflow
    workflow = GroupChatBuilder(
        participants=[coder_agent, answer_composer_agent],
        orchestrator_agent=orchestrator_agent,
        max_rounds=20,
    ).build()
    
    return client, workflow


async def run_chat(workflow, prompt: str) -> List[Dict[str, Any]]:
    """
    Run the workflow for a given prompt and collect detailed event logs.
    
    Args:
        workflow: The workflow instance to use
        prompt: The user prompt to process
        
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
    
    try:
        current_agent = None
        # Track current message being accumulated from streaming tokens
        current_message = None
        current_source = None
        last_tool_log_index = 0
        
        # Run the workflow with streaming
        async for event in workflow.run(prompt, stream=True):
            event_type = type(event).__name__
            
            # Track which agent is currently active
            if hasattr(event, 'executor_id') and event.executor_id:
                current_agent = event.executor_id
            
            # Check if there are new tool logs to inject
            if last_tool_log_index < len(_tool_call_logs):
                # Save current accumulated message before inserting tool logs
                if current_message is not None:
                    detailed_logs.append(current_message)
                    current_message = None
                    current_source = None
                
                # Insert new tool logs
                detailed_logs.extend(_tool_call_logs[last_tool_log_index:])
                last_tool_log_index = len(_tool_call_logs)
            
            # Map Agent Framework events to the expected log format
            if hasattr(event, 'text') and event.text and current_agent:
                # Accumulate streaming tokens into complete messages
                if current_source == current_agent and current_message is not None:
                    current_message["data"]["content"] += event.text
                else:
                    # Save previous message if it exists
                    if current_message is not None:
                        detailed_logs.append(current_message)
                    
                    # Start new message
                    current_message = {
                        "type": "TextMessage",
                        "data": {
                            "source": current_agent,
                            "content": event.text
                        }
                    }
                    current_source = current_agent
            
            # Check for tool call events
            if hasattr(event, 'tool_calls') and event.tool_calls:
                # Save current accumulated message first
                if current_message is not None:
                    detailed_logs.append(current_message)
                    current_message = None
                    current_source = None
                
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
                        "source": current_agent or "unknown",
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
                    current_source = None
                
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
                        "source": current_agent or "unknown",
                        "tool_call_results": tool_results
                    }
                }
                detailed_logs.append(log_entry)
        
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
    client, workflow = create_agent_team()
    
    try:
        # Demo message
        message = "List me all unpaid sales invoices for the last year"
        
        print(f"\n--- Running Demo 2 (Agent Framework) standalone for: '{message}' ---")
        
        # Run with the workflow
        async for event in workflow.run(message, stream=True):
            if hasattr(event, 'text') and event.text:
                agent_name = getattr(event, 'executor_id', 'unknown')
                print(f"\n[{agent_name}]: {event.text}", end="", flush=True)
        
        print("\n\n--- Demo complete ---")
        
    finally:
        print("--- Cleanup complete ---")


if __name__ == "__main__":
    asyncio.run(main())
