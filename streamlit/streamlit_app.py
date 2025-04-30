import asyncio
import streamlit as st
import sys  # Import sys to check the platform
import os
import importlib.util
import json
import re
import ast # Re-import ast

# --- Configuration ---
DEMO_CONFIG = {
    "demo1": {
        "name": "Demo 1: Code + Executor",
        "path": os.path.join("demo-1. bc-api-agent-2-way-code", "v0.4", "main.py"),
        "prompts": ["List me the Cash CoA"],
        "caption": "Using AssistantAgent + CodeExecutorAgent (2 Agents)"
    },
    "demo2": {
        "name": "Demo 2: Coder + Executor + Composer",
        "path": os.path.join("demo-2. bc-api-agent-group-code", "v0.4", "main.py"),
        "prompts": ["List me all unpaid sales invoices for the last year"],
         "caption": "Using SelectorGroupChat (3 Agents, LLM selects next)"
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

    if not os.path.exists(main_path):
        st.error(f"Could not find main.py for {config['name']} at: {main_path}")
        return None, None

    try:
        spec = importlib.util.spec_from_file_location(f"main_module_{demo_id}", main_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.create_agent_team, module.run_chat
    except Exception as e:
        st.error(f"Failed to load module for {config['name']}: {e}")
        return None, None

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
    selected_demo_id = st.sidebar.selectbox(
        "Choose a Demo:",
        options=list(demo_options.keys()),
        format_func=lambda demo_id: demo_options[demo_id],
        key="selected_demo_selector", # Use a distinct key for the widget
        index=list(demo_options.keys()).index(st.session_state.get("selected_demo", DEFAULT_DEMO)) # Set default index
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
    current_config = DEMO_CONFIG[st.session_state.selected_demo]
    st.title(current_config["name"])
    st.caption(current_config["caption"])

    # --- Load Demo Modules ---
    # We load these every time, but only initialize agents if needed
    create_agent_team_func, run_chat_func = load_demo_module(st.session_state.selected_demo)
    if not create_agent_team_func or not run_chat_func:
        st.error("Failed to load necessary functions for the selected demo. Please check paths and files.")
        st.stop()

    # --- Example Prompts / Pills ---
    st.markdown("##### Try an example:")
    example_prompts = current_config["prompts"]
    cols = st.columns(len(example_prompts))
    for i, example_prompt in enumerate(example_prompts):
        # Use demo_id in the key to ensure buttons are unique across demos
        if cols[i].button(example_prompt, key=f"example_{st.session_state.selected_demo}_{i}"):
            reset_chat_state() # Reset before starting new chat with example
            st.session_state.messages.append({"role": "user", "content": example_prompt})
            st.rerun()

    # --- Initialize State (Messages & Agents) ---
    if "messages" not in st.session_state:
         st.session_state.messages = []

    # Initialize agent components *only* if they don't exist for the current session
    if "agent_components" not in st.session_state:
        try:
            # Call the dynamically loaded create_agent_team function
            # Unpack carefully, only store what's needed universally (group_chat)
            # Demo 1 returns 4 items, Demo 2 returns 5. We only need group_chat.
            returned_items = create_agent_team_func()
            model_client = returned_items[0] # Assume model_client is always first
            group_chat = returned_items[-1] # Assume group_chat is always last

            st.session_state["agent_components"] = {
                # "model_client": model_client, # Optional: Store if needed e.g. for cleanup
                "group_chat": group_chat
            }
        except Exception as e:
            st.error(f"Failed to initialize agent team for {current_config['name']}: {e}")
            if "agent_components" in st.session_state: del st.session_state.agent_components # Clean up partial init
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

        with st.spinner(f"Processing with {current_config['name']}..."):
            try:
                # Use the dynamically loaded run_chat function
                response_messages = asyncio.run(run_chat_func(group_chat, user_prompt))

                # Process responses (same logic as before)
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
                    # Heuristic for final answer (might need adjustment depending on demo's last speaker)
                    if role in ["bc_api_agent", "executor_agent", "Answer_Composer_Agent"]: # Cover both demos
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