import asyncio
import streamlit as st
import sys
import os
import json
import re
import ast
import traceback
import time
import importlib

# --- Set Page Config (Must be the FIRST Streamlit command) ---
st.set_page_config(page_title="AutoGen BC API Demos", page_icon="🤖", layout="wide")

# --- Path Setup for Imports ---
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


def load_demo_module(demo_id: str, framework: str):
    """
    Dynamically load a demo module using importlib.
    
    Args:
        demo_id: The demo ID (e.g., 'demo1')
        framework: The framework type ('autogen' or 'agent-framework')
        
    Returns:
        Tuple of (create_agent_team function, run_chat function) or (None, None) on error
    """
    try:
        # Determine version path based on framework
        if framework == "autogen":
            version_path = "v0.4"
        else:  # agent-framework
            version_path = "v1.0"
        
        # Build the full path to the version directory
        module_dir = os.path.join(workspace_root, demo_id, framework, version_path)
        
        if not os.path.exists(module_dir):
            st.error(f"Module directory not found: {module_dir}")
            return None, None
        
        # Add the module directory to sys.path if not already there
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        # Clear cached demo modules to prevent cross-contamination between
        # frameworks/demos that share module names (e.g., bc_api_tool exists in
        # both autogen and agent-framework versions with different decorators)
        demo_modules_to_clear = ['main', 'bc_api_tool']
        for mod_name in demo_modules_to_clear:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        
        main_module = importlib.import_module('main')
        
        # Extract the functions
        create_func = getattr(main_module, 'create_agent_team', None)
        run_func = getattr(main_module, 'run_chat', None)
        
        if not create_func or not run_func:
            st.error(f"Required functions not found in {demo_id}/{framework}/{version_path}/main.py")
            return None, None
        
        return create_func, run_func
        
    except ImportError as e:
        st.error(f"Failed to import {demo_id} ({framework}): {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None
    except Exception as e:
        st.error(f"Error loading {demo_id} ({framework}): {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

# --- Configuration ---
DEMO_CONFIG = {
    "demo1": {
        "name": "Demo 1 (Simple Agent): Code + Executor",
        "prompts": ["Show me cash accounts"],
        "captions": {
            "autogen": "Autogen v0.4: AssistantAgent + CodeExecutorAgent (2 Agents)",
            "agent-framework": "Agent Framework v1.0: Single Agent with execute_python_code tool"
        }
    },
    "demo2": {
        "name": "Demo 2 (Team of Agents): Manager + Coder + Executor + Composer",
        "prompts": ["Show me cash accounts"],
        "captions": {
            "autogen": "Autogen v0.4: SelectorGroupChat (3 Agents, LLM selects next)",
            "agent-framework": "Agent Framework v1.0: GroupChatBuilder (2 Agents + Orchestrator)"
        }
    },
    "demo3": {
        "name": "Demo 3: Agent + Tool",
        "prompts": ["Which customers owes money to the company?"],
        "captions": {
            "autogen": "Autogen v0.4: AssistantAgent with custom Tool (1 Agent)",
            "agent-framework": "Agent Framework v1.0: Single Agent with @tool decorator"
        }
    },
    "demo4": {
        "name": "Demo 4: Agent + RAG + Tool",
        "prompts": ["Which customers owes money to the company?"],
        "captions": {
            "autogen": "Autogen v0.4: AssistantAgent with RAG and API Tool (1 Agent)",
            "agent-framework": "Agent Framework v1.0: Single Agent with RAG + @tool"
        }
    }
}
DEFAULT_DEMO = "demo1"
DEFAULT_FRAMEWORK = "autogen"

# --- Constants for Final Answer Agent Names ---
FINAL_ANSWER_AGENTS = {
    "demo1": "bc_api_agent",
    "demo2": "Answer_Composer_Agent",
    "demo3": "BC_API_Assistant",
    "demo4": "BC_API_Assistant",
}

# --- Constants for Demo Images (relative to this app.py file) ---
DEMO_IMAGES = {
    "demo1": "../demo1/autogen/v0.4/demo1.png",
    "demo2": "../demo2/autogen/v0.4/demo2.png",
    "demo3": "../demo3/autogen/v0.4/demo3.png",
    "demo4": "../demo4/autogen/v0.4/demo4.png",
}

# --- Utility Functions (load_demo_module removed) ---

# Set the asyncio policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

def get_demo_functions(demo_id, framework):
    """
    Returns the create_agent_team and run_chat functions for the selected demo and framework.
    
    Args:
        demo_id: The demo ID (e.g., 'demo1')
        framework: The framework type ('autogen' or 'agent-framework')
        
    Returns:
        Tuple of (create_func, run_func) or (None, None) on error
    """
    return load_demo_module(demo_id, framework)

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
            
            # Define threshold for long messages (500 characters)
            MESSAGE_LENGTH_THRESHOLD = 500
            
            # Check if message is long
            is_long_message = len(content) > MESSAGE_LENGTH_THRESHOLD
            
            if is_long_message:
                # Create a unique key for this message based on content hash
                import hashlib
                message_key = f"show_full_{hashlib.md5(content.encode()).hexdigest()[:8]}"
                
                # Initialize session state for this message if not exists
                if message_key not in st.session_state:
                    st.session_state[message_key] = False
                
                # Show truncated version
                truncated_content = content[:MESSAGE_LENGTH_THRESHOLD] + "..."
                if "```" in truncated_content:
                    st.markdown(truncated_content, unsafe_allow_html=True)
                else:
                    st.markdown(truncated_content)
                
                # Button to toggle full message view with proper state sync
                # Read current state first
                current_state = st.session_state[message_key]
                button_label = "Hide full message" if current_state else "Show full message"
                
                # Use on_click callback to ensure state updates before rerun
                def toggle_message():
                    st.session_state[message_key] = not st.session_state[message_key]
                
                st.button(button_label, key=f"btn_{message_key}", use_container_width=False, on_click=toggle_message)
                
                # Show full message if toggled
                if st.session_state[message_key]:
                    st.markdown("---")
                    if "```" in content:
                        st.markdown(content, unsafe_allow_html=True)
                    else:
                        # Use text_area for better readability of long text (non-code)
                        st.text_area("Full message:", value=content, height=300, disabled=True, key=f"full_{message_key}", label_visibility="collapsed")
            else:
                # Show full content for short messages
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
                tool_name = tc.get('name', 'unknown_tool')
                st.markdown(f"* **Tool:** `{tool_name}` (ID: `{tc.get('id')}`)")
                try:
                    # Attempt to parse arguments if they are a JSON string
                    args_dict = json.loads(tc.get('arguments', '{}'))
                    
                    # Special handling for execute_python_code tool - show code nicely
                    if tool_name == 'execute_python_code' and 'code' in args_dict:
                        st.markdown("**Generated Python Code:**")
                        st.code(args_dict['code'], language='python')
                    else:
                        # For other tools, show JSON
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
                      # Special handling for execute_python_code - show as execution output
                      if tool_name == 'execute_python_code':
                          if isinstance(content, str):
                              # Show execution output in a code block
                              if len(content) > 500:
                                  st.text_area("Execution Output:", value=content, height=200, disabled=True)
                              else:
                                  st.code(content, language='text')
                          else:
                              st.text(f"Result: {content}")
                      else:
                          # For other tools, try to parse as JSON first
                          try:
                             content_data = json.loads(content)
                             st.json(content_data)
                          except (json.JSONDecodeError, TypeError):
                              if isinstance(content, str) and len(content) > 500:
                                  st.text_area("Result Content:", value=content, height=200, disabled=True)
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
            
            # Extract content property if it exists
            content_value = log_data.get("content", None)
            other_properties = {k: v for k, v in log_data.items() if k != "content"}
            
            # Display content property prominently (always shown)
            if content_value is not None:
                st.markdown("**Content:**")
                try:
                    # Try to display as JSON if it's a dict/list
                    if isinstance(content_value, (dict, list)):
                        st.json(content_value)
                    else:
                        # If it's a string, try to parse as JSON first
                        try:
                            parsed = json.loads(content_value) if isinstance(content_value, str) else content_value
                            st.json(parsed)
                        except (json.JSONDecodeError, TypeError):
                            # Display as text if not JSON
                            if isinstance(content_value, str) and len(content_value) > 500:
                                import hashlib
                                content_key = f"content_{log_type}_{hashlib.md5(str(content_value).encode()).hexdigest()[:8]}"
                                st.text_area("Content:", value=content_value, height=200, disabled=True, key=content_key, label_visibility="collapsed")
                            else:
                                st.code(str(content_value), language='text')
                except Exception:
                    st.text(f"Content: {content_value}")
            
            # Display other properties with button to show/hide (hidden by default)
            if other_properties:
                import hashlib
                other_props_key = f"show_other_props_{log_type}_{hashlib.md5(str(log_data).encode()).hexdigest()[:8]}"
                
                # Initialize session state if not exists (default to False = hidden)
                if other_props_key not in st.session_state:
                    st.session_state[other_props_key] = False
                
                # Button to toggle other properties visibility
                current_state = st.session_state[other_props_key]
                button_label = "Hide other properties" if current_state else "Show other properties"
                
                def make_toggle(key):
                    def toggle():
                        st.session_state[key] = not st.session_state[key]
                    return toggle
                
                st.button(button_label, key=f"btn_{other_props_key}", use_container_width=False, on_click=make_toggle(other_props_key))
                
                # Show other properties if toggled
                if st.session_state[other_props_key]:
                    st.markdown("**Other Properties:**")
                    try:
                        st.json(other_properties)
                    except Exception:
                        for prop_key, prop_value in other_properties.items():
                            st.text(f"**{prop_key}:** {prop_value}")
            
            # If no content property and no other properties, show all data
            if content_value is None and not other_properties:
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
    # --- Display Toast from previous run (if any) ---
    if 'toast_summary' in st.session_state:
        st.toast(st.session_state['toast_summary'], icon="📊")
        del st.session_state['toast_summary'] # Clear after showing
        
    # This function is now called after st.set_page_config()
    
    # --- Sidebar for Demo Selection ---
    st.sidebar.title("Demo Selection")
    
    # Framework is fixed to Autogen v0.4 (selector hidden)
    selected_framework = st.session_state.get("selected_framework", DEFAULT_FRAMEWORK)
    if "selected_framework" not in st.session_state:
        st.session_state.selected_framework = DEFAULT_FRAMEWORK
    
    # Demo selector
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

    # --- Display Demo Image in Sidebar --- 
    image_relative_path = DEMO_IMAGES.get(selected_demo_id)
    if image_relative_path:
        # Construct full path relative to the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.normpath(os.path.join(script_dir, image_relative_path))
        if os.path.exists(image_path):
            st.sidebar.image(image_path, use_container_width=True)
        else:
            st.sidebar.warning(f"Image file not found: {image_path}")
    # --- End Display Demo Image in Sidebar ---

    # --- Handle Demo Change ---
    demo_changed = st.session_state.get("selected_demo") != selected_demo_id
    
    if demo_changed:
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
    if selected_demo_id not in DEMO_CONFIG:
        st.error(f"Selected demo '{selected_demo_id}' is not configured.")
        st.stop()
    current_config = DEMO_CONFIG[selected_demo_id]
    st.title(current_config["name"])
    
    # Get caption based on selected framework
    caption = current_config["captions"].get(selected_framework, "")
    st.caption(caption)
    
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
        # Only load demo module when we need to create the agent (avoids unnecessary
        # module reloads on every Streamlit rerun and prevents module cache contamination)
        create_agent_team_func, run_chat_func = get_demo_functions(selected_demo_id, selected_framework)
        if not create_agent_team_func or not run_chat_func:
            st.error(f"Failed to load functions for demo '{selected_demo_id}' with framework '{selected_framework}'. Check if files exist.")
            st.stop()
        try:
            returned_items = create_agent_team_func()
            if returned_items is None:
                st.error("create_agent_team_func returned None. Check environment variables and module setup.")
                st.stop()
            group_chat = returned_items[-1] # Assume group_chat is always last
            if group_chat is None:
                st.error("Group chat is None. Check the create_agent_team function.")
                st.stop()
            # Store run_chat_func from the SAME module load to keep tool log references in sync
            st.session_state["agent_components"] = {"group_chat": group_chat, "run_chat_func": run_chat_func}
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
    prompt = st.chat_input(f"Ask about Business Central data ({selected_demo_id})...")
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
        selected_framework = st.session_state.get("selected_framework", DEFAULT_FRAMEWORK)

        if "agent_components" not in st.session_state or "group_chat" not in st.session_state.agent_components:
            st.warning("Agent components not initialized. Please refresh or select a demo.")
            st.stop()
        group_chat = st.session_state.agent_components["group_chat"]
        # Use the run_chat_func stored from the SAME module load as create_agent_team
        # This ensures tool log globals (_tool_call_logs) are shared between tools and run_chat
        run_chat_func = st.session_state.agent_components.get("run_chat_func")
        
        if not run_chat_func:
            st.error(f"Could not find run_chat function. Please refresh the page.")
            st.stop()
        
        # Create live streaming display with st.status
        with st.chat_message("assistant", avatar="🤖"):
            status = st.status("Processing...", expanded=True)
            text_placeholder = status.empty()
            accumulated_text = ""
            current_agent = None
            
            def handle_event(event):
                nonlocal accumulated_text, text_placeholder, current_agent
                etype = event.get("type")
                
                if etype == "streaming_token":
                    source = event.get("source")
                    if source != current_agent:
                        # Finalize previous text block
                        if accumulated_text:
                            text_placeholder.markdown(accumulated_text)
                        current_agent = source
                        accumulated_text = ""
                        status.markdown(f"**{source}** is thinking...")
                        text_placeholder = status.empty()
                    accumulated_text += event.get("content", "")
                    text_placeholder.markdown(accumulated_text + " ▌")
                
                elif etype == "tool_call":
                    if accumulated_text:
                        text_placeholder.markdown(accumulated_text)
                        accumulated_text = ""
                    tool_name = event.get("tool_name", "unknown")
                    status.markdown(f"🛠️ **Calling** `{tool_name}`")
                    args = event.get("arguments", {})
                    if tool_name == "execute_python_code" and "code" in args:
                        status.code(args["code"], language="python")
                    elif tool_name == "query_qdrant_documentation" and "query" in args:
                        status.markdown(f"📚 Query: {args['query']}")
                    else:
                        status.json(args)
                    text_placeholder = status.empty()
                
                elif etype == "tool_result":
                    is_error = event.get("is_error", False)
                    tool_name = event.get("tool_name", "unknown")
                    if is_error:
                        status.error(f"❌ Tool error: {tool_name}")
                        content_str = str(event.get("content", ""))[:500]
                        status.code(content_str, language="text")
                    else:
                        status.success(f"✅ {tool_name} completed")
                        # Show truncated preview for non-code execution tools
                        if tool_name != "execute_python_code":
                            content_str = str(event.get("content", ""))
                            if len(content_str) > 300:
                                status.markdown(f"*Result preview:* {content_str[:300]}...")
                    text_placeholder = status.empty()
                
                elif etype == "agent_change":
                    agent_name = event.get("agent_name", "unknown")
                    status.markdown(f"---\n**🔄 Agent selected:** {agent_name}")
                    current_agent = agent_name
                    accumulated_text = ""
                    text_placeholder = status.empty()
            
            try:
                detailed_logs = asyncio.run(run_chat_func(group_chat, user_prompt, on_event=handle_event))
                
                # Finalize accumulated text
                if accumulated_text:
                    text_placeholder.markdown(accumulated_text)
                status.update(label="Processing complete", state="complete", expanded=False)

                # --- Analyze logs for summary --- 
                agent_turns = 0
                tool_calls = 0
                tool_errors = 0
                for log in detailed_logs:
                    # We only count TextMessages from non-user sources as agent turns
                    if log.get("type") == "TextMessage" and log.get("data", {}).get("source") != "user": 
                        agent_turns += 1
                    elif log.get("type") == "ToolCallRequestMessage":
                        tool_calls += len(log.get("data", {}).get("tool_calls", []))
                    elif log.get("type") == "ToolCallResultMessage":
                        for result in log.get("data", {}).get("tool_call_results", []):
                            if result.get("is_error"): 
                                tool_errors += 1
                
                summary_msg = f"Processing complete: {agent_turns} agent message(s)"
                if tool_calls > 0:
                     summary_msg += f", {tool_calls} tool call(s)"
                if tool_errors > 0:
                     summary_msg += f" ({tool_errors} error(s))"
                # Store summary for toast on next run instead of showing immediately
                st.session_state['toast_summary'] = summary_msg 
                # --- End of summary logic ---
                
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
                status.update(label="Error occurred", state="error", expanded=True)
                st.error(f"An error occurred during processing: {e}")
                st.error(f"Traceback: {traceback.format_exc()}")
                st.session_state["internal_messages"] = [{"role": "system", "content": f"Error: {e}"}]
                st.rerun()

    # --- Display Internal Messages Expander --- Filtered
    if "internal_messages" in st.session_state:
        with st.expander("View Processing Details", expanded=False):
            for message in st.session_state.internal_messages:
                # Skip displaying the TaskResult event
                if message.get("type") == "TaskResult":
                    continue 
                display_message(message)

if __name__ == "__main__":
    main() 