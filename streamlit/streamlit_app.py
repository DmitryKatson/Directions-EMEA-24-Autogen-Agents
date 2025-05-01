import asyncio
import streamlit as st
import sys  # Import sys to check the platform
import os
import importlib.util
import json
import re
import ast # Re-import ast
import traceback # Add for better error diagnostics

# --- Configuration ---
DEMO_CONFIG = {
    "demo1": {
        "name": "Demo 1 (Simple Agent): Code + Executor",
        "path": os.path.join("demo-1. bc-api-agent-2-way-code", "v0.4", "main.py"),
        "prompts": ["List me the Cash CoA"],
        "caption": "Using AssistantAgent + CodeExecutorAgent (2 Agents)"
    },
    "demo2": {
        "name": "Demo 2 (Team of Agents): Manager + Coder + Executor + Composer",
        "path": os.path.join("demo-2. bc-api-agent-group-code", "v0.4", "main.py"),
        "prompts": ["List me all unpaid sales invoices for the last year"],
         "caption": "Using SelectorGroupChat (3 Agents, LLM selects next)"
    },
    "demo3": {
        "name": "Demo 3: Agent + Tool",
        "path": os.path.join("demo-3. bc-api-agent-2-way-tool", "v0.4", "main.py"),
        "prompts": ["Which customers owes money to the company?"],
        "caption": "Using AssistantAgent with a custom Tool (1 Agent)"
    }
}
DEFAULT_DEMO = "demo1"

# --- Utility Functions ---

# Set the asyncio policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

def load_demo_module(demo_id):
    """Dynamically loads the main module for the selected demo."""
    config = DEMO_CONFIG.get(demo_id)
    if not config:
        st.error(f"Invalid demo ID: {demo_id}")
        return None, None

    # Construct path relative to the workspace root (one level up from this script's dir)
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_path = os.path.join(workspace_root, config["path"])
    main_dir = os.path.dirname(main_path)

    if not os.path.exists(main_path):
        st.error(f"Could not find main.py for {config['name']} at: {main_path}")
        return None, None

    try:
        module_name = f"main_module_{demo_id}"
        # Invalidate cache if module already loaded
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        # Add the directory containing main.py to the Python path temporarily
        # This ensures that imports inside main.py can find modules in the same directory
        original_sys_path = sys.path.copy()
        sys.path.insert(0, main_dir)
        
        # Also handle any potential module name conflicts by clearing them from sys.modules
        bc_tool_name = "bc_api_tool"
        if bc_tool_name in sys.modules:
            del sys.modules[bc_tool_name]
        
        spec = importlib.util.spec_from_file_location(module_name, main_path)
        if spec is None:
            st.error(f"Could not create module spec for {config['name']} at: {main_path}")
            return None, None
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # Add to sys.modules before exec
                
        spec.loader.exec_module(module)
        
        # Check if required functions exist
        if not hasattr(module, 'create_agent_team') or not hasattr(module, 'run_chat'):
            st.error(f"Module {config['name']} loaded, but missing create_agent_team or run_chat function.")
            return None, None
            
        # Store module functions before restoring path
        create_agent_team = module.create_agent_team
        run_chat = module.run_chat
        
        # Restore the original Python path
        sys.path = original_sys_path
        
        return create_agent_team, run_chat
    except ImportError as e:
        st.error(f"Import error loading module for {config['name']}: {e}. Check dependencies.")
        # Display more detailed diagnostics for bc_api_tool specifically
        if "bc_api_tool" in str(e):
            tool_path = os.path.join(main_dir, "bc_api_tool.py")
            if os.path.exists(tool_path):
                st.info(f"bc_api_tool.py exists at {tool_path}, but failed to import.")
            else:
                st.error(f"bc_api_tool.py not found at {tool_path}")
        return None, None
    except Exception as e:
        st.error(f"Failed to load module for {config['name']}: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        # Clean up partially loaded module if error occurred during exec
        if module_name in sys.modules:
            del sys.modules[module_name]
        return None, None
    finally:
        # Ensure path is restored even if an exception occurs
        if 'original_sys_path' in locals():
            sys.path = original_sys_path

def display_message(message):
    """Displays messages, attempting to format Python literals or JSON within text."""
    role = message.get("role", "unknown")
    content = message.get("content", "")

    with st.chat_message(role):
        # Handle existing markdown code blocks first
        if "```" in content:
            st.markdown(content, unsafe_allow_html=True)
            return

        if role == "user":
             st.markdown(content)
             return

        # Attempt to find and format Python literal / JSON
        match = re.search(r'(.*?)(\[.*?\]|\{.*?\})', content, re.DOTALL)
        if match:
            prefix = match.group(1).strip()
            literal_str = match.group(2)
            parsed_data = None
            try: # Prioritize ast.literal_eval
                parsed_data = ast.literal_eval(literal_str)
            except: # Fallback to JSON conversion
                try:
                    json_str = literal_str.replace("'", '"')
                    json_str = re.sub(r'\bNone\b', 'null', json_str)
                    json_str = re.sub(r'\bTrue\b', 'true', json_str)
                    json_str = re.sub(r'\bFalse\b', 'false', json_str)
                    parsed_data = json.loads(json_str)
                except:
                    parsed_data = None

            if parsed_data is not None:
                if prefix: st.markdown(prefix)
                st.json(parsed_data)
                return

        # Fallback for everything else
        st.markdown(content)

def reset_chat_state():
    """Clears messages and agent components from session state."""
    st.session_state.messages = []
    if "internal_messages" in st.session_state:
        del st.session_state["internal_messages"]
    if "agent_components" in st.session_state:
        del st.session_state["agent_components"] # Remove old components

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="AutoGen BC API Demos", page_icon="🤖", layout="wide")

    # --- Sidebar for Demo Selection ---
    st.sidebar.title("Demo Selection")
    # Use demo names for display, store demo ID
    demo_options = {demo_id: config["name"] for demo_id, config in DEMO_CONFIG.items()}

    # Determine default index safely
    default_index = 0
    current_selection = st.session_state.get("selected_demo", DEFAULT_DEMO)
    if current_selection in demo_options:
        default_index = list(demo_options.keys()).index(current_selection)

    selected_demo_id = st.sidebar.selectbox(
        "Choose a Demo:",
        options=list(demo_options.keys()),
        format_func=lambda demo_id: demo_options.get(demo_id, "Unknown Demo"), # Handle potential key errors
        key="selected_demo_selector", # Use a distinct key for the widget
        index=default_index # Set default index
    )

    # --- Handle Demo Change ---
    # Check if the selection changed compared to the stored state
    if st.session_state.get("selected_demo") != selected_demo_id:
        st.session_state.selected_demo = selected_demo_id
        reset_chat_state() # Reset state when demo changes
        st.rerun() # Rerun immediately to load new demo config/prompts

    # Ensure selected_demo is initialized in state if it wasn't already
    if "selected_demo" not in st.session_state:
         st.session_state.selected_demo = selected_demo_id # Usually DEFAULT_DEMO on first run

    # --- Load Config for Selected Demo ---
    # Check if selected_demo_id is valid before accessing DEMO_CONFIG
    if st.session_state.selected_demo not in DEMO_CONFIG:
         st.error(f"Selected demo '{st.session_state.selected_demo}' is not configured.")
         st.stop()
    current_config = DEMO_CONFIG[st.session_state.selected_demo]
    st.title(current_config["name"])
    st.caption(current_config["caption"])

    # --- Load Demo Modules ---
    # We load these every time, but only initialize agents if needed
    create_agent_team_func, run_chat_func = load_demo_module(st.session_state.selected_demo)
    if not create_agent_team_func or not run_chat_func:
        # Error is already displayed by load_demo_module
        st.stop()

    # --- Example Prompts / Pills ---
    st.markdown("##### Try an example:")
    example_prompts = current_config["prompts"]
    # Ensure at least one column, even if no prompts are defined
    num_cols = max(1, len(example_prompts))
    cols = st.columns(num_cols)
    for i, example_prompt in enumerate(example_prompts):
        # Use demo_id in the key to ensure buttons are unique across demos
        if cols[i % num_cols].button(example_prompt, key=f"example_{st.session_state.selected_demo}_{i}"):
            reset_chat_state() # Reset before starting new chat with example
            st.session_state.messages = [{"role": "user", "content": example_prompt}] # Start with only this message
            st.rerun()

    # --- Initialize State (Messages & Agents) ---
    if "messages" not in st.session_state:
         st.session_state.messages = []

    # Initialize agent components *only* if they don't exist for the current session
    if "agent_components" not in st.session_state:
        try:
            # Call the dynamically loaded create_agent_team function
            # Use a try-except block with improved error handling
            try:
                # Demo specific detailed error handling
                if st.session_state.selected_demo == "demo3":
                    # For Demo 3, first check if bc_api_tool.py exists
                    demo_dir = os.path.dirname(os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        current_config["path"]
                    ))
                    tool_path = os.path.join(demo_dir, "bc_api_tool.py")
                    if not os.path.exists(tool_path):
                        st.error(f"bc_api_tool.py not found at {tool_path}. "
                                 f"Please ensure this file exists before running Demo 3.")
                        st.stop()
                
                # Now try to initialize the agents
                returned_items = create_agent_team_func()
                
                # Unpack carefully, only store what's needed universally (group_chat)
                # Demo 1 returns 4, Demo 2 returns 5, Demo 3 returns 3 items.
                # model_client = returned_items[0] # Assume model_client is always first (Optional)
                group_chat = returned_items[-1] # Assume group_chat is always last

                st.session_state["agent_components"] = {
                    "group_chat": group_chat
                }
            except TypeError as e:
                if "args" in str(e) and st.session_state.selected_demo == "demo3":
                    # Special handling for this specific error in Demo 3
                    st.error(f"Error initializing Demo 3: {e}. This might be due to an issue with bc_api_tool.py. "
                             f"Please ensure the file exists and has the correct structure.")
                else:
                    # General TypeError handling
                    st.error(f"Type error initializing agents: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                if "agent_components" in st.session_state: 
                    del st.session_state.agent_components
                st.stop()
            except ImportError as e:
                if "bc_api_tool" in str(e) and st.session_state.selected_demo == "demo3":
                    # Special handling for bc_api_tool import error
                    st.error(f"Failed to import bc_api_tool: {e}. "
                             f"Please ensure bc_api_tool.py exists in the Demo 3 directory.")
                else:
                    st.error(f"Import error initializing agents: {e}")
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
        group_chat = st.session_state.agent_components["group_chat"]

        with st.spinner(f"Processing..."):
            try:
                # Use the dynamically loaded run_chat function
                response_messages = asyncio.run(run_chat_func(group_chat, user_prompt))

                # Process responses
                internal_messages_processed = []
                final_answer_content = "No response generated."
                for i, msg in enumerate(response_messages):
                    role = "system"; content = "Error getting content"
                    try:
                        if hasattr(msg.source, 'name') and isinstance(msg.source.name, str): role = msg.source.name
                        elif isinstance(msg.source, str): role = msg.source
                    except AttributeError: role = "unknown_source"
                    try: content = msg.to_text() if hasattr(msg, 'to_text') else str(msg)
                    except Exception as content_e: content = f"Error getting content for message {i}: {content_e}"
                    internal_messages_processed.append({"role": role, "content": content})

                    # --- Updated Final Answer Heuristic ---
                    # Check agent names from all demos for the final answer
                    final_answer_agents = ["bc_api_agent", "executor_agent", "Answer_Composer_Agent", "BC_API_Assistant"]
                    if role in final_answer_agents:
                        final_answer_content = content

                # Store results
                st.session_state.messages.append({"role": "assistant", "content": final_answer_content})
                st.session_state["internal_messages"] = internal_messages_processed
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.session_state["internal_messages"] = [{"role": "system", "content": f"Error: {e}"}]
                st.rerun()

    # --- Display Internal Messages Expander ---
    if "internal_messages" in st.session_state:
        with st.expander("View Processing Details", expanded=False):
            for message in st.session_state.internal_messages:
                display_message(message)

if __name__ == "__main__":
    main() 