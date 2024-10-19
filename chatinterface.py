import os
import sqlite3
import requests
import json
import logging
import shelve
import gradio as gr
import uuid
from datetime import datetime

# Define constants
OLLAMA_URL = "http://localhost:11434/api/chat"  # Replace with your Ollama endpoint if different
MODEL_NAME = "llama3.1:8b-instruct-fp16"        # The model version
CHARACTER_DIR = 'characters'                    # Directory where character text files are stored

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)  # Use a named logger

# Ensure the 'characters' directory exists
def ensure_character_directory():
    """Ensure the 'characters' directory exists, create it if not, and add a default character."""
    if not os.path.exists(CHARACTER_DIR):
        os.makedirs(CHARACTER_DIR)
        logger.info(f"Created '{CHARACTER_DIR}' directory.")
        default_character_file = os.path.join(CHARACTER_DIR, 'default_character.txt')
        with open(default_character_file, 'w', encoding='utf-8') as f:
            f.write("You are a friendly, supportive assistant with a cheerful personality.")
        logger.info(f"Added default character file: {default_character_file}")

# SQLite DB functions for long-term memory storage
def init_db():
    """Initialize SQLite database for storing conversation summaries and histories."""
    conn = sqlite3.connect('memory.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS sessions
                      (session_id TEXT PRIMARY KEY,
                       character_file TEXT,
                       history TEXT,
                       summary TEXT)''')
    conn.commit()
    conn.close()

def store_session_data(session_id, character_file, history, summary):
    """Store session data including character file, history, and summary in SQLite database."""
    conn = sqlite3.connect('memory.db')
    cursor = conn.cursor()
    # Convert history to JSON string
    history_json = json.dumps(history)
    cursor.execute('REPLACE INTO sessions (session_id, character_file, history, summary) VALUES (?, ?, ?, ?)',
                   (session_id, character_file, history_json, summary))
    conn.commit()
    conn.close()

def retrieve_session_data(session_id):
    """Retrieve session data including character file, history, and summary from SQLite database."""
    conn = sqlite3.connect('memory.db')
    cursor = conn.cursor()
    cursor.execute('SELECT character_file, history, summary FROM sessions WHERE session_id = ?', (session_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        character_file = row[0]
        history_json = row[1]
        summary = row[2]
        history = json.loads(history_json) if history_json else []
        return character_file, history, summary
    else:
        return None, [], ""

def delete_session_data(session_id):
    """Delete session data from SQLite database."""
    conn = sqlite3.connect('memory.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()

# Cache initialization using shelve
def init_cache():
    """Initialize the shelve-based cache for persistent storage."""
    cache = shelve.open('chat_cache', writeback=True)  # Persistent cache stored in 'chat_cache'
    return cache

def close_cache(cache):
    """Close the shelve cache when no longer needed."""
    if cache is not None:
        cache.close()

def generate_cache_key(user_prompt: str, memory: str, system_prompt: str, model: str) -> str:
    """
    Generate a unique cache key based on the user prompt, memory, system prompt, and model.
    Including 'memory' ensures that the cache is specific to the conversation context.
    """
    return f"{model}:{system_prompt}:{memory}:{user_prompt}"

# Updated LLM API interaction with proper response handling and correct payload format
def generate_response_with_llm(user_prompt: str, memory: str, system_prompt: str, model: str) -> str:
    """Call the LLM via API to generate responses with caching and proper payload format."""
    cache = init_cache()
    cache_key = generate_cache_key(user_prompt, memory, system_prompt, model)

    # Check if the result is already cached
    if cache_key in cache:
        logger.info(f"Fetching result from cache for prompt: {user_prompt[:50]}...")
        response_content = cache[cache_key]
        close_cache(cache)
        return response_content

    # If not cached, call the LLM API
    try:
        # Construct the full message history for the LLM
        messages = [
            {"role": "system", "content": system_prompt},  # System prompt
            {"role": "user", "content": user_prompt}       # User prompt
        ]
        if memory:
            messages.insert(1, {"role": "assistant", "content": memory})  # Insert memory if available

        logger.info(f"Sending request to LLM with model '{model}' and prompt size {len(user_prompt)}")

        payload = {
            "model": model,
            "messages": messages  # Proper structure
        }

        logger.debug(f"Payload: {json.dumps(payload)}")

        headers = {'Content-Type': 'application/json'}
        # Enable streaming
        response = requests.post(OLLAMA_URL, data=json.dumps(payload), headers=headers, stream=True)

        logger.debug(f"Response status code: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"Failed to generate response with LLM: HTTP {response.status_code}")
            logger.debug(f"Response content: {response.text}")
            close_cache(cache)
            return ""

        # Read the streaming response
        response_content = ""
        raw_response = []
        for line in response.iter_lines():
            if line:
                try:
                    line_decoded = line.decode('utf-8')
                    raw_response.append(line_decoded)  # Store the raw line for debugging
                    data = json.loads(line_decoded)
                    # Check if 'content' field is present in the message
                    if 'message' in data and 'content' in data['message']:
                        response_content += data['message']['content']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError as e:
                    logger.error(f"JSONDecodeError: {e}")
                    logger.debug(f"Line content: {line}")
                    continue

        if not response_content.strip():
            logger.warning("Received empty response from LLM.")
            logger.debug(f"Complete raw response: {raw_response}")
            close_cache(cache)
            return ""

        # Cache the result
        logger.debug("Caching the generated response.")
        cache[cache_key] = response_content
        cache.sync()
        close_cache(cache)

        return response_content.strip()

    except Exception as e:
        logger.error(f"Failed to generate response with LLM: {e}")
        close_cache(cache)
        return ""

# Functions for Gradio UI
def get_character_files():
    """Retrieve a list of character files from the character directory."""
    return [f for f in os.listdir(CHARACTER_DIR) if f.endswith('.txt')]

def load_character_prompt(character_file):
    """Load the system prompt from the selected character file."""
    with open(os.path.join(CHARACTER_DIR, character_file), 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    return system_prompt

def get_existing_sessions(character_file):
    """Retrieve a list of existing session IDs for the selected character."""
    conn = sqlite3.connect('memory.db')
    cursor = conn.cursor()
    cursor.execute('SELECT session_id FROM sessions WHERE character_file = ?', (character_file,))
    rows = cursor.fetchall()
    conn.close()
    session_ids = [row[0] for row in rows]
    logger.debug(f"Existing sessions for character '{character_file}': {session_ids}")
    return session_ids

def get_character_name(character_file):
    """Extract character name from character file name."""
    character_name = os.path.splitext(character_file)[0]
    return character_name

def add_name_to_history(history, character_name):
    """Add 'name' field to assistant messages in the history and update content to include character name."""
    for message in history:
        if message['role'] == 'assistant':
            message['name'] = character_name
            # Replace "Assistant: " with character_name + ": " in content
            if "Assistant: " in message['content']:
                message['content'] = message['content'].replace("Assistant: ", f"{character_name}: ")
    return history

def update_character_info(character_file):
    """Generate the content for the character info display."""
    character_name = get_character_name(character_file)
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    info_text = f"**Talking to:** {character_name} | **Current Time:** {current_time}"
    return gr.update(value=info_text)

def on_character_change(character_file):
    """Handle changes in character selection."""
    logger.debug(f"Character changed to: {character_file}")
    # Update session dropdown choices
    existing_sessions = get_existing_sessions(character_file)
    logger.debug(f"Updated session list: {existing_sessions}")
    # Load the system prompt
    system_prompt = load_character_prompt(character_file)
    # Reset chat history and memory
    history = []
    memory = ""
    # Update character info display
    character_info = update_character_info(character_file)
    # Return updates
    return (
        gr.update(choices=existing_sessions, value=None),  # session_id_dropdown
        gr.update(value=[]),                              # chatbot
        [],                                               # history
        "",                                               # memory
        system_prompt,                                    # system_prompt
        character_info                                    # character_info_display
    )

def on_session_change(character_file, session_id):
    """Handle changes in session ID."""
    logger.debug(f"Character: {character_file}, Session ID: {session_id}")
    if not session_id:
        # If no session is selected, clear the history and memory
        history = []
        memory = ""
        system_prompt = load_character_prompt(character_file)
        character_info = update_character_info(character_file)
        return (
            gr.update(value=[]),  # chatbot
            history,              # history
            memory,               # memory
            system_prompt,        # system_prompt
            character_info        # character_info_display
        )

    # Load session data
    stored_character_file, history, summary = retrieve_session_data(session_id)
    if stored_character_file is None:
        # New session, initialize history and summary
        history = []
        summary = ""
        # Store initial session data
        store_session_data(session_id, character_file, history, summary)
    else:
        # If the character file in the session data is different, update it
        if stored_character_file != character_file:
            store_session_data(session_id, character_file, history, summary)
    # Load system prompt
    system_prompt = load_character_prompt(character_file)
    # Get current time for info display
    character_info = update_character_info(character_file)
    # Add 'name' to assistant messages and update content
    character_name = get_character_name(character_file)
    history = add_name_to_history(history, character_name)
    # Return history, summary (as memory), system_prompt, and character info
    return (
        gr.update(value=history),  # chatbot
        history,                    # history
        summary,                    # memory
        system_prompt,              # system_prompt
        character_info              # character_info_display
    )

def create_new_session(character_file):
    """Create a new session ID and update the session dropdown."""
    logger.debug(f"Creating new session for character: {character_file}")
    # Generate a new session ID
    new_session_id = str(uuid.uuid4())
    # Initialize session data
    history = []
    memory = ""
    # Load system prompt
    system_prompt = load_character_prompt(character_file)
    # Store initial session data
    store_session_data(new_session_id, character_file, history, memory)
    # Update session dropdown choices
    existing_sessions = get_existing_sessions(character_file)
    logger.debug(f"Updated session list after creation: {existing_sessions}")
    # Update character info display
    character_info = update_character_info(character_file)
    # Return the updated session dropdown and select the new session
    return (
        gr.update(choices=existing_sessions, value=new_session_id),  # session_id_dropdown
        gr.update(value=[]),                                        # chatbot
        [],                                                         # history
        "",                                                         # memory
        system_prompt,                                              # system_prompt
        character_info                                              # character_info_display
    )

def delete_session(character_file, session_id):
    """Delete the selected session and update the session dropdown."""
    logger.debug(f"Deleting session ID: {session_id} for character: {character_file}")
    if session_id:
        delete_session_data(session_id)
    # Update session dropdown choices
    existing_sessions = get_existing_sessions(character_file)
    logger.debug(f"Updated session list after deletion: {existing_sessions}")
    # Reset chat history and memory
    history = []
    memory = ""
    # Load system prompt
    system_prompt = load_character_prompt(character_file)
    # Update character info display
    character_info = update_character_info(character_file)
    # Return the updated session dropdown
    return (
        gr.update(choices=existing_sessions, value=None),  # session_id_dropdown
        gr.update(value=[]),                              # chatbot
        [],                                               # history
        "",                                               # memory
        system_prompt,                                    # system_prompt
        character_info                                    # character_info_display
    )

def respond(user_input, history, memory, system_prompt, session_id, character_file):
    """Generate a response from the model and update the chat history."""
    logger.debug(f"User input: {user_input}")
    # Check for empty user input
    if not user_input.strip():
        logger.warning("Received empty user input.")
        return (
            gr.update(value=history),  # chatbot
            history,                    # history
            memory,                     # memory
            system_prompt,              # system_prompt
            gr.update(value='')         # user_input
        )
    
    if not session_id:
        logger.warning("No session ID selected.")
        return (
            gr.update(value=history),  # chatbot
            history,                    # history
            memory,                     # memory
            system_prompt,              # system_prompt
            gr.update(value='')         # user_input
        )
    
    # Ensure system_prompt and memory are loaded
    if not system_prompt:
        system_prompt = load_character_prompt(character_file)
    if memory is None:
        _, _, memory = retrieve_session_data(session_id)
    
    # Generate the response from the LLM
    response = generate_response_with_llm(user_input, memory, system_prompt, MODEL_NAME)
    
    if not response:
        response = "I'm sorry, I couldn't process your request. Please try again."
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%I:%M %p")
    
    # Extract character name
    character_name = get_character_name(character_file)
    
    # Append to the chat history as dicts for 'messages' type with timestamps and character name
    history = history + [
        {"role": "user", "content": f"[{timestamp}] {user_input}"},
        {"role": "assistant", "content": f"[{timestamp}] {response}", "name": character_name}
    ]
    
    # Update memory (summary) if necessary
    memory += f"\n{response}"
    
    # Store the updated session data
    store_session_data(session_id, character_file, history, memory)
    
    # Return updated history to both chatbot and history state
    return (
        gr.update(value=history),  # chatbot
        history,                    # history
        memory,                     # memory
        system_prompt,              # system_prompt
        gr.update(value='')         # user_input
    )

def reset_chat(session_id, character_file):
    """Reset the chat history and memory without deleting the session."""
    logger.debug(f"Resetting chat for session ID: {session_id}, character: {character_file}")
    # Clear the chat history and memory
    history = []
    memory = ""
    # Load system prompt
    system_prompt = load_character_prompt(character_file)
    # Update the session data to reset history and memory
    if session_id:
        store_session_data(session_id, character_file, history, memory)
    # Update character info display
    character_info = update_character_info(character_file)
    return (
        gr.update(value=[]),  # chatbot
        history,              # history
        memory,               # memory
        system_prompt,        # system_prompt
        gr.update(value=''), # user_input
        character_info        # character_info_display
    )

def main():
    # Initialize the SQLite DB
    init_db()

    # Ensure character directory exists and has default character
    ensure_character_directory()

    # Get the list of character files
    character_files = get_character_files()

    if not character_files:
        logger.error(f"No character files found in '{CHARACTER_DIR}' directory.")
        return

    # Set default character file
    default_character_file = character_files[0]

    with gr.Blocks() as demo:
        gr.Markdown("# Chatbot with Character Selection and Session Management")

        # Character info display
        character_info_display = gr.Markdown()

        # Initialize session variables
        session_id = gr.State(None)
        history = gr.State([])
        memory = gr.State("")
        system_prompt = gr.State(load_character_prompt(default_character_file))

        with gr.Row():
            character_dropdown = gr.Dropdown(
                choices=character_files,
                label="Select Character",
                value=default_character_file
            )
            session_id_dropdown = gr.Dropdown(
                choices=[],
                label="Select Session",
                value=None
            )
            new_session_button = gr.Button("Create New Session")
            delete_session_button = gr.Button("Delete Session")

        # Initialize Chatbot with 'messages' type
        chatbot = gr.Chatbot(value=[], type='messages')
        user_input = gr.Textbox(label="Your Message", placeholder="Type your message here and press Enter")

        # Update session dropdown and load session data when character changes
        character_dropdown.change(
            on_character_change,
            inputs=character_dropdown,
            outputs=[session_id_dropdown, chatbot, history, memory, system_prompt, character_info_display]
        )

        # Load session data when session ID changes
        session_id_dropdown.change(
            on_session_change,
            inputs=[character_dropdown, session_id_dropdown],
            outputs=[chatbot, history, memory, system_prompt, character_info_display]
        )

        # Handle user input submission
        user_input.submit(
            respond,
            inputs=[user_input, history, memory, system_prompt, session_id_dropdown, character_dropdown],
            outputs=[chatbot, history, memory, system_prompt, user_input]
        )

        # Create new session
        new_session_button.click(
            create_new_session,
            inputs=character_dropdown,
            outputs=[session_id_dropdown, chatbot, history, memory, system_prompt, character_info_display]
        )

        # Delete session
        delete_session_button.click(
            delete_session,
            inputs=[character_dropdown, session_id_dropdown],
            outputs=[session_id_dropdown, chatbot, history, memory, system_prompt, character_info_display]
        )

        # Add a reset button
        reset_button = gr.Button("Reset Chat")
        reset_button.click(
            reset_chat,
            inputs=[session_id_dropdown, character_dropdown],
            outputs=[chatbot, history, memory, system_prompt, user_input, character_info_display]
        )

        # On app load, load existing sessions and set default session if available
        def load_default_session():
            character_file = character_dropdown.value
            existing_sessions = get_existing_sessions(character_file)
            logger.debug(f"Loading default session for character: {character_file}")
            if existing_sessions:
                selected_session_id = existing_sessions[0]
                stored_character_file, loaded_history, loaded_memory = retrieve_session_data(selected_session_id)
                system_prompt_value = load_character_prompt(stored_character_file)
            else:
                # If no existing sessions, create a new one
                selected_session_id = str(uuid.uuid4())
                store_session_data(selected_session_id, character_file, [], "")
                existing_sessions = [selected_session_id]
                loaded_history = []
                loaded_memory = ""
                system_prompt_value = load_character_prompt(character_file)
                logger.debug(f"Created new session: {selected_session_id}")
            # Update character info display
            character_info = update_character_info(character_file)
            # Update components
            return (
                gr.update(choices=existing_sessions, value=selected_session_id),  # session_id_dropdown
                gr.update(value=loaded_history),                                  # chatbot
                loaded_history,                                                  # history state
                loaded_memory,                                                   # memory state
                system_prompt_value,                                            # system_prompt state
                character_info                                                   # character_info_display
            )

        # Call load_default_session after the interface is launched
        demo.load(load_default_session, inputs=None, outputs=[session_id_dropdown, chatbot, history, memory, system_prompt, character_info_display])

    # Launch Gradio with share=True if you want a public link
    demo.launch(share=False)  # Set to True if you want to create a public link

if __name__ == "__main__":
    main()
