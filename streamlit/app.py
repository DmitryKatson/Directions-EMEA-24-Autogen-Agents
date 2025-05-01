import asyncio
import streamlit as st
import sys
import os
import json
import re
import ast
import traceback

# --- Set Page Config (Must be the FIRST Streamlit command) ---
st.set_page_config(page_title="AutoGen BC API Demos", page_icon="🤖", layout="wide")

# --- Path Setup for Imports ---
# Add workspace root to path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Explicitly add demo subdirectories to path to help resolve relative imports within them
demo3_v04_path = os.path.join(workspace_root, "demo3", "v04")
if demo3_v04_path not in sys.path:
    sys.path.insert(0, demo3_v04_path)
    
demo4_v04_path = os.path.join(workspace_root, "demo4", "v04")
if demo4_v04_path not in sys.path:
    sys.path.insert(0, demo4_v04_path)

# --- Import Demo Functions Directly ---
try:
    from demo1.v04.main import create_agent_team as create_demo1_team, run_chat as run_demo1_chat
except ImportError as e:
    st.error(f"Failed to import Demo 1: {e}. Ensure demo1/v04/main.py exists and is importable.")
    create_demo1_team, run_demo1_chat = None, None

try:
    from demo2.v04.main import create_agent_team as create_demo2_team, run_chat as run_demo2_chat
except ImportError as e:
    st.error(f"Failed to import Demo 2: {e}. Ensure demo2/v04/main.py exists and is importable.")
    create_demo2_team, run_demo2_chat = None, None

try:
    # Now that demo3/v04 is in the path, the relative import inside main.py should work
    from demo3.v04.main import create_agent_team as create_demo3_team, run_chat as run_demo3_chat
except ImportError as e:
    st.error(f"Failed to import Demo 3: {e}. Ensure demo3/v04/main.py and demo3/v04/bc_api_tool.py exist and are importable.")
    create_demo3_team, run_demo3_chat = None, None

try:
    # Now that demo4/v04 is in the path, the relative import inside main.py should work
    from demo4.v04.main import create_agent_team as create_demo4_team, run_chat as run_demo4_chat
except ImportError as e:
    st.error(f"Failed to import Demo 4: {e}. Ensure demo4/v04/main.py and demo4/v04/bc_api_tool.py exist and are importable.")
    create_demo4_team, run_demo4_chat = None, None

# --- Configuration (Paths are no longer needed) ---
DEMO_CONFIG = {
    "demo1": {
        "name": "Demo 1 (Simple Agent): Code + Executor",
        "prompts": ["List me the Cash CoA"],
        "caption": "Using AssistantAgent + CodeExecutorAgent (2 Agents)",
        "create_func": create_demo1_team,
        "run_func": run_demo1_chat
    },
    "demo2": {
        "name": "Demo 2 (Team of Agents): Manager + Coder + Executor + Composer",
        "prompts": ["List me all unpaid invoices since the beginning of the year"],
        "caption": "Using SelectorGroupChat (3 Agents, LLM selects next)",
        "create_func": create_demo2_team,
        "run_func": run_demo2_chat
    },
    "demo3": {
        "name": "Demo 3: Agent + Tool",
        "prompts": ["Which customers owes money to the company?"],
        "caption": "Using AssistantAgent with a custom Tool (1 Agent)",
        "create_func": create_demo3_team,
        "run_func": run_demo3_chat
    },
    "demo4": {
        "name": "Demo 4: Agent + RAG + Tool",
        "prompts": ["Which customers owes money to the company?"],
        "caption": "Using AssistantAgent with RAG and API Tool (1 Agent)",
        "create_func": create_demo4_team,
        "run_func": run_demo4_chat
    }
}
DEFAULT_DEMO = "demo1"

# --- Constants for Final Answer Agent Names ---
FINAL_ANSWER_AGENTS = {
    "demo1": "bc_api_agent",
    "demo2": "Answer_Composer_Agent",
    "demo3": "BC_API_Assistant",
    "demo4": "BC_API_Assistant",
}

# --- Utility Functions (load_demo_module removed) ---

# Set the asyncio policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

def get_demo_functions(demo_id):
    """Returns the create_agent_team and run_chat functions for the selected demo."""
    config = DEMO_CONFIG.get(demo_id)
    if not config:
        st.error(f"Invalid demo ID: {demo_id}")
        return None, None

    create_func = config.get('create_func')
    run_func = config.get('run_func')
    
    if not create_func or not run_func:
        st.error(f"Functions not loaded for Demo ID: {demo_id}. Check imports.")
        return None, None
        
    return create_func, run_func

def display_message(message_or_log):
    """Displays messages (simple format) or detailed log entries (new format)."""
    
    # Check if it's the simple message format
    if "role" in message_or_log and "content" in message_or_log:
        # --- Original logic for simple messages --- 
        role = message_or_log.get("role", "unknown")
        content = message_or_log.get("content", "")
        icon = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=icon):
            if "```" in content:
                st.markdown(content, unsafe_allow_html=True)
                return
            # Removed the complex literal/JSON parsing for simplicity, 
            # similar to how it was handled in the new log logic for TextMessage
            st.markdown(content)
        return # Handled simple message
        # --- End of original logic ---
        
    # Otherwise, assume it's the detailed log entry format
    log_entry = message_or_log
    log_type = log_entry.get("type", "unknown")
    log_data = log_entry.get("data", {})

    role = "system" # Default role for logs
    icon = "⚙️" # Default icon for logs

    # --- New logic for detailed logs --- 
    if log_type == "user_prompt":
        role = "user"
        icon = "🧑"
        with st.chat_message(role, avatar=icon):
            st.markdown(f"**Log: User Prompt**")
            st.markdown(log_entry.get("content", "[No content]"))

    elif log_type == "TextMessage":
        role = log_data.get("source", "assistant")
        icon = "🤖" if role != "user" else "🧑" # Keep consistent icons
        content = log_data.get("content", "")
        with st.chat_message(role, avatar=icon):
            st.markdown(f"**Log: Agent Message ({role})**")
            if "```" in content:
                 st.markdown(content, unsafe_allow_html=True)
            else:
                 st.markdown(content)

    elif log_type == "ToolCallRequestMessage":
        source_agent = log_data.get("source", "tool_request")
        role = source_agent # Use agent name as role
        icon = "🛠️"
        with st.chat_message(role, avatar=icon):
            st.markdown(f"**Log: Tool Call Request ({role})**")
            tool_calls = log_data.get("tool_calls", [])
            if not tool_calls: st.markdown("_(No tool calls found in data)_")
            for tc in tool_calls:
                st.markdown(f"* **Tool:** `{tc.get('name', 'unknown_tool')}` (ID: `{tc.get('id')}`)")
                try:
                    # Attempt to parse arguments if they are a JSON string
                    args_dict = json.loads(tc.get('arguments', '{}')) 
                    st.code(json.dumps(args_dict, indent=2), language='json')
                except (json.JSONDecodeError, TypeError):
                     # Otherwise, display raw arguments
                    st.text(f"Arguments: {tc.get('arguments', {})}")
        
    elif log_type == "ToolCallResultMessage":
        role = "tool_result" # Use a specific role for results
        tool_results = log_data.get("tool_call_results", [])
        if not tool_results: 
             with st.chat_message(role, avatar="ℹ️"):
                  st.markdown("_(No tool results found in data)_")
                  return

        for result in tool_results:
             is_error = result.get("is_error", False)
             tool_name = result.get("name", "unknown_tool")
             call_id = result.get("call_id")
             content = result.get("content", "")
             icon = "❌" if is_error else "✅" # Error or Success icon
             
             with st.chat_message(role, avatar=icon):
                 if is_error:
                     st.error(f"**Log: Tool Error ({tool_name} - ID: {call_id})**")
                     try:
                         error_data = json.loads(content)
                         st.json(error_data)
                     except (json.JSONDecodeError, TypeError):
                         st.code(content, language='text')
                 else:
                      st.success(f"**Log: Tool Result ({tool_name} - ID: {call_id})**")
                      try:
                         content_data = json.loads(content)
                         st.json(content_data)
                      except (json.JSONDecodeError, TypeError):
                          if isinstance(content, str) and len(content) > 500:
                              st.text_area("Result Content:", value=content, height=200)
                          elif isinstance(content, str):
                              st.markdown(f"```\n{content}\n```")
                          else:
                              st.text(f"Result: {content}") # Fallback for non-string content
        
    else:
        # Fallback for other known or unknown log event types
        role = log_type # Use the type name as role
        icon = "⚙️"
        with st.chat_message(role, avatar=icon):
            st.markdown(f"**Log: Event Type:** `{log_type}`")
            try:
                st.json(log_data) 
            except Exception:
                 st.text(f"Data: {log_data}") # Fallback if data isn't JSON serializable
    # --- End of new logic ---

def reset_chat_state():
    """Clears messages and agent components from session state."""
    st.session_state.messages = []
    if "internal_messages" in st.session_state:
        del st.session_state["internal_messages"]
    if "agent_components" in st.session_state:
        del st.session_state["agent_components"]

# Function to show the embedding UI for Demo 4
def show_embedding_ui():
    """Shows the embedding UI for Demo 4 to index documents."""
    st.header("Document Embedding for RAG")
    
    st.markdown("""
    ### What is RAG (Retrieval Augmented Generation)?
    RAG enhances AI responses by combining a language model with a knowledge base. When you ask a question:
    1. The system searches for relevant information in the indexed documents
    2. It combines this retrieved information with its general knowledge
    3. It generates a more accurate, contextual response
    
    ### How to use this feature:
    1. Upload a PDF document (like an API manual or business documentation)
    2. Click "Index Document" to process and embed the content
    3. Once indexed, you can ask specific questions about the document content
    """)
    
    # Removed st.columns to simplify layout after removing env check
    
    # File uploader for PDF documents
    uploaded_file = st.file_uploader("Upload a PDF document for indexing", type="pdf")
    
    default_pdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "demo4", "embedding", "Business_Central_API_v2.0_Reference_2nd_Edition.pdf")
    
    use_default_doc = st.checkbox("Use default Business Central API documentation",
                                 value=os.path.exists(default_pdf_path))
    
    if use_default_doc:
        st.info("Using the default Business Central API v2.0 Reference documentation")
        file_to_index = default_pdf_path
        can_index = True
    elif uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "demo4", "embedding", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File saved: {uploaded_file.name}")
        file_to_index = temp_file_path
        can_index = True
    else:
        can_index = False
    
    # Environment check section removed
    
    # Button to start the embedding process
    if can_index:
        if st.button("Index Document"):
            # Use st.progress with callback for detailed updates
            progress_bar = st.progress(0, text="Starting indexing...")
            
            # Define the callback function for progress updates
            def update_progress(step, total_steps, description):
                progress = (step / total_steps) if total_steps > 0 else 0
                progress_bar.progress(progress, text=f"{description} ({step}/{total_steps})")
                
            try:
                # Import the embedding module dynamically
                # Ensure the embedding directory itself is in the path
                embedding_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo4", "embedding")
                if embedding_dir not in sys.path:
                    sys.path.insert(0, embedding_dir) # Add to front to prioritize
                
                # Now import the function
                from embedding import process_and_embed_pdf
                
                # Process and embed the PDF - pass the callback function
                process_and_embed_pdf(file_to_index, progress_callback=update_progress)
                
                # Update progress bar to 100% on completion
                progress_bar.progress(100, text="Indexing Complete!")
                st.success("Document indexing complete!") # Keep success message
                
            except ImportError as ie:
                progress_bar.empty() # Remove progress bar on error
                st.error(f"Import Error during embedding: {ie}. Check if 'embedding.py' exists in demo4/embedding.")
                st.error(traceback.format_exc())
            except Exception as e:
                progress_bar.empty() # Remove progress bar on error
                st.error(f"Error during indexing: {e}")
                st.error(traceback.format_exc())
    else:
        st.info("Please upload a document or use the default documentation to begin indexing.")

# --- Main Streamlit App --- 
def main():
    # This function is now called after st.set_page_config()
    
    # --- Sidebar for Demo Selection ---
    st.sidebar.title("Demo Selection")
    demo_options = {demo_id: config["name"] for demo_id, config in DEMO_CONFIG.items()}
    default_index = 0
    current_selection = st.session_state.get("selected_demo", DEFAULT_DEMO)
    if current_selection in demo_options:
        default_index = list(demo_options.keys()).index(current_selection)
    selected_demo_id = st.sidebar.selectbox(
        "Choose a Demo:",
        options=list(demo_options.keys()),
        format_func=lambda demo_id: demo_options.get(demo_id, "Unknown Demo"),
        key="selected_demo_selector",
        index=default_index
    )

    # --- Handle Demo Change ---
    if st.session_state.get("selected_demo") != selected_demo_id:
        st.session_state.selected_demo = selected_demo_id
        reset_chat_state()
        st.rerun()

    if "selected_demo" not in st.session_state:
         st.session_state.selected_demo = selected_demo_id

    # --- Show Embedding UI for Demo 4 in Sidebar ---
    if st.session_state.selected_demo == "demo4":
        with st.sidebar.expander("Document Embedding (RAG)", expanded=True):
            show_embedding_ui()

    # --- Load Config for Selected Demo ---
    if st.session_state.selected_demo not in DEMO_CONFIG:
         st.error(f"Selected demo '{st.session_state.selected_demo}' is not configured.")
         st.stop()
    current_config = DEMO_CONFIG[st.session_state.selected_demo]
    st.title(current_config["name"])
    st.caption(current_config["caption"])

    # --- Get Demo Functions (using direct imports) ---
    create_agent_team_func, run_chat_func = get_demo_functions(st.session_state.selected_demo)
    if not create_agent_team_func or not run_chat_func:
        st.error(f"Failed to load functions for demo '{st.session_state.selected_demo}'. Check imports at the top of the script.")
        st.stop()

    # --- Example Prompts / Pills ---
    st.markdown("##### Try an example:")
    example_prompts = current_config["prompts"]
    num_cols = max(1, len(example_prompts))
    cols = st.columns(num_cols)
    for i, example_prompt in enumerate(example_prompts):
        if cols[i % num_cols].button(example_prompt, key=f"example_{st.session_state.selected_demo}_{i}"):
            reset_chat_state()
            st.session_state.messages = [{"role": "user", "content": example_prompt}]
            st.rerun()

    # --- Initialize State (Messages & Agents) ---
    if "messages" not in st.session_state:
         st.session_state.messages = []

    if "agent_components" not in st.session_state:
        try:
            returned_items = create_agent_team_func()
            if returned_items is None:
                st.error("create_agent_team_func returned None. Check environment variables and module setup.")
                st.stop()
            group_chat = returned_items[-1] # Assume group_chat is always last
            if group_chat is None:
                st.error("Group chat is None. Check the create_agent_team function.")
                st.stop()
            st.session_state["agent_components"] = {"group_chat": group_chat}
        except ImportError as e:
             st.error(f"Import error during agent initialization for {current_config['name']}: {e}")
             st.error(f"Traceback: {traceback.format_exc()}")
             st.warning("Ensure the workspace root directory is in your Python path or run Streamlit from the root.")
             if "agent_components" in st.session_state: 
                 del st.session_state.agent_components
             st.stop()
        except Exception as e:
            st.error(f"Failed to initialize agent team for {current_config['name']}: {e}")
            st.error(f"Traceback: {traceback.format_exc()}")
            if "agent_components" in st.session_state: 
                del st.session_state.agent_components
            st.stop()

    # --- Display Chat History ---
    for message in st.session_state.get("messages", []):
        display_message(message)

    # --- Chat Input ---
    prompt = st.chat_input(f"Ask about Business Central data ({st.session_state.selected_demo})...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        if "internal_messages" in st.session_state:
            del st.session_state["internal_messages"]
        st.rerun()

    # --- Process New User Messages ---
    if (st.session_state.get("messages") and
            st.session_state.messages[-1]["role"] == "user" and
            "internal_messages" not in st.session_state):
        user_prompt = st.session_state.messages[-1]["content"]
        selected_demo_id = st.session_state.get("selected_demo", DEFAULT_DEMO)

        if "agent_components" not in st.session_state or "group_chat" not in st.session_state.agent_components:
            st.warning("Agent components not initialized. Please refresh or select a demo.")
            st.stop()
        group_chat = st.session_state.agent_components["group_chat"]
        with st.spinner(f"Processing..."):
            try:
                # Get the detailed logs from the run_chat function
                run_chat_func = get_demo_functions(selected_demo_id)[1] # Get the specific run_chat func
                if not run_chat_func:
                     st.error(f"Could not find run_chat function for demo {selected_demo_id}")
                     st.stop()
                     
                detailed_logs = asyncio.run(run_chat_func(group_chat, user_prompt))

                # Extract the final assistant message for main display - using dynamic agent name
                final_answer_content = "No final answer found in logs."
                expected_final_agent = FINAL_ANSWER_AGENTS.get(selected_demo_id, "assistant") # Fallback
                
                for log in reversed(detailed_logs):
                    log_type = log.get("type")
                    log_data = log.get("data", {})
                    source_agent = log_data.get("source")
                    
                    if log_type == "TextMessage" and source_agent == expected_final_agent:
                        final_answer_content = log_data.get("content", final_answer_content)
                        # Optional: Remove termination signal if present for cleaner display
                        if final_answer_content.strip().endswith(("TERMINATE", "TASK_COMPLETE")): 
                             final_answer_content = re.sub(r'\s*(TERMINATE|TASK_COMPLETE)\s*$','', final_answer_content).strip()
                        break # Found the last relevant message
                
                st.session_state.messages.append({"role": "assistant", "content": final_answer_content})
                
                # Store the raw detailed logs for the expander
                st.session_state["internal_messages"] = detailed_logs 
                
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.error(f"Traceback: {traceback.format_exc()}")
                st.session_state["internal_messages"] = [{"role": "system", "content": f"Error: {e}"}]
                st.rerun()

    # --- Display Internal Messages Expander ---
    if "internal_messages" in st.session_state:
        with st.expander("View Processing Details", expanded=False):
            for message in st.session_state.internal_messages:
                display_message(message)

if __name__ == "__main__":
    main() 