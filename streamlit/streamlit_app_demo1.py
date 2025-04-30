import asyncio
import streamlit as st
import sys  # Import sys to check the platform
import os
import importlib.util
import json
import re
import ast # Re-import ast

# Set the asyncio policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Directly import from the module path
# Go up one level from the current file's directory (streamlit) to the workspace root, then construct the path
workspace_root = os.path.dirname(os.path.dirname(__file__))
main_path = os.path.join(workspace_root, "demo-1. bc-api-agent-2-way-code", "v0.4", "main.py")
# Check if the path exists for debugging
if not os.path.exists(main_path):
    print(f"ERROR: Calculated main_path does not exist: {main_path}") # Add a check
    # Optionally raise an error or handle it appropriately
    # raise FileNotFoundError(f"Could not find main.py at {main_path}")
spec = importlib.util.spec_from_file_location("main_module", main_path)
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)
create_agent_team = main_module.create_agent_team
run_chat = main_module.run_chat

def display_message(message):
    """Displays messages, attempting to format Python literals or JSON within text."""
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        # Handle existing markdown code blocks first
        if "```" in content:
            st.markdown(content, unsafe_allow_html=True)
            return

        if role == "user":
             st.markdown(content)
             return

        # --- Attempt to find and format Python literal / JSON ---
        # Regex to find a structure starting with { or [ potentially after some text
        match = re.search(r'(.*?)(\[.*?\]|\{.*?\})', content, re.DOTALL)

        if match:
            prefix = match.group(1).strip()
            literal_str = match.group(2) # The potential dict/list string
            parsed_data = None

            # 1. Try parsing as Python literal using ast.literal_eval (safer for Python dicts/lists)
            try:
                # ast.literal_eval safely evaluates a string containing a Python literal
                parsed_data = ast.literal_eval(literal_str)
            except (ValueError, SyntaxError, MemoryError, TypeError):
                 # If ast fails, proceed to try JSON conversion
                 parsed_data = None
            except Exception: # Catch any other unexpected errors from ast
                 parsed_data = None


            # 2. If ast failed, try converting to JSON format and parsing
            if parsed_data is None:
                try:
                    # Convert Python bool/None and single quotes to JSON standard
                    json_str = literal_str.replace("'", '"')
                    json_str = re.sub(r'\bNone\b', 'null', json_str)
                    json_str = re.sub(r'\bTrue\b', 'true', json_str)
                    json_str = re.sub(r'\bFalse\b', 'false', json_str)
                    # Parse the converted string as JSON
                    parsed_data = json.loads(json_str)
                except Exception:
                    # If JSON conversion/parsing also fails, give up
                    parsed_data = None

            # If parsing succeeded using either method, display prefix + formatted data
            if parsed_data is not None:
                if prefix:
                    st.markdown(prefix)
                # Use st.json for built-in formatting of the parsed data
                st.json(parsed_data)
                return # Stop processing, we're done

        # --- Fallback ---
        # If no structure found by regex, or if all parsing attempts failed
        st.markdown(content)

def main():
    st.set_page_config(page_title="BC API Agent Demo", page_icon="🤖")
    st.title("BC API Agent Demo")
    st.caption("Business Central API Code Generation Assistant")

    # --- Example Prompts / Pills ---
    example_prompts = ["List me the Cash CoA"]
    cols = st.columns(len(example_prompts))
    for i, example_prompt in enumerate(example_prompts):
        if cols[i].button(example_prompt, key=f"example_{i}"):
            # Add the example prompt to messages and trigger rerun
            st.session_state.messages.append({"role": "user", "content": example_prompt})
            if "internal_messages" in st.session_state:
                del st.session_state["internal_messages"]
            st.rerun()
    # --- End Example Prompts ---


    # Initialize agent components and chat history
    if "agent_components" not in st.session_state:
        try:
            model_client, bc_api_agent, executor_agent, group_chat = create_agent_team()
            st.session_state["agent_components"] = {
                "model_client": model_client,
                "group_chat": group_chat
            }
        except Exception as e:
            st.error(f"Failed to initialize agent team: {e}")
            st.stop()

        # Initialize messages only if agent components are successfully initialized
        if "messages" not in st.session_state:
             st.session_state["messages"] = []

    # Display chat history
    for message in st.session_state.get("messages", []):
        display_message(message)

    # Chat input
    prompt = st.chat_input("Ask about Business Central data...") # Updated placeholder text
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        if "internal_messages" in st.session_state:
            del st.session_state["internal_messages"]
        st.rerun()

    # Process new messages if the last message was from the user
    if (st.session_state.get("messages") and
            st.session_state.messages[-1]["role"] == "user" and
            "internal_messages" not in st.session_state):

        user_prompt = st.session_state.messages[-1]["content"]
        group_chat = st.session_state.agent_components["group_chat"]

        with st.spinner("Processing..."):
            try:
                # Run the chat
                response_messages = asyncio.run(run_chat(group_chat, user_prompt))

                # Process responses
                internal_messages_processed = []
                final_answer_content = "No response generated."

                for i, msg in enumerate(response_messages):
                    role = "system"
                    content = "Error getting content"
                    try:
                        if hasattr(msg.source, 'name') and isinstance(msg.source.name, str):
                            role = msg.source.name
                        elif isinstance(msg.source, str):
                             role = msg.source
                    except AttributeError:
                        role = "unknown_source"

                    try:
                        content = msg.to_text() if hasattr(msg, 'to_text') else str(msg)
                    except Exception as content_e:
                         content = f"Error getting content for message {i}: {content_e}"

                    internal_messages_processed.append({"role": role, "content": content})

                    if role in ["bc_api_agent", "executor_agent"]:
                        final_answer_content = content

                # Store results
                st.session_state.messages.append({"role": "assistant", "content": final_answer_content})
                st.session_state["internal_messages"] = internal_messages_processed
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.session_state["internal_messages"] = [{"role": "system", "content": f"Error: {e}"}]
                st.rerun()


    # Display internal messages in an expander
    if "internal_messages" in st.session_state:
        with st.expander("View Processing Details", expanded=False):
            for message in st.session_state.internal_messages:
                display_message(message)

if __name__ == "__main__":
    main() 