import os
import sqlite3
import requests
import json
import logging
import shelve
import gradio as gr
import uuid
from datetime import datetime
import hashlib
import re
import traceback
import threading
import time
import queue
import atexit
import yaml
from logging.handlers import RotatingFileHandler
from typing import List

# Define constants
OLLAMA_URL = "http://localhost:11434/api/chat"  # Replace with your Ollama endpoint if different
MODEL_NAME = "llama3.1:70b-instruct-q4_K_M"
CHECK_MODEL_NAME = "llama3.1:70b-instruct-q4_K_M"
CHARACTER_DIR = 'characters'  # Directory where character text files are stored
LOG_FILE = 'app.log'  # Log file path
# Configurable number of retries for LLM requests
MAX_RETRIES = 5  # Set the desired number of retries

# Summarization settings (configurable)
MAX_CONTEXT_LENGTH = 32000  # Max context length before summarizing
DEFAULT_NUMBER_OF_RECENT_MESSAGES_TO_KEEP = 20  # Configurable number of recent messages to keep in history after summarizing
# Number of history lines to consider in response validation (configurable)
DEFAULT_NUMBER_OF_HISTORY_LINES_FOR_VALIDATION = 3
SUMMARY_PROMPT_TEMPLATE = (
    "Please provide a concise but comprehensive summary of the following conversation, "
    "including all important details, timestamps, and topics discussed:\n{conversation}\nSummary:"
)

# Initialize a lock for session data to prevent race conditions
session_lock = threading.Lock()

# Clear root logger's handlers to avoid duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up the root logger to handle warnings and above for other libraries
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)

# Set up your application logger
logger = logging.getLogger('my_app')
logger.setLevel(logging.DEBUG)

# Clear any existing handlers to prevent duplicates
if logger.hasHandlers():
    logger.handlers.clear()

# File handler for your app's log file
file_handler = RotatingFileHandler(LOG_FILE, mode='w', maxBytes=10**8, backupCount=0)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Console handler for debugging output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add handlers to your app's logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Global cache variables
cache = None
cache_lock = threading.RLock()  # Use RLock for reentrant locking

# Add this utility function to remove '/skip' tokens
def remove_skip_tokens(text):
    """Remove '/skip' tokens from the text."""
    return re.sub(r'/skip\s*$', '', text).strip()

def is_first_assistant_message(history):
    assistant_messages = [msg for msg in history if msg['role'] == 'assistant']
    return len(assistant_messages) == 0

# Initialize cache using shelve
def init_cache():
    """Initialize the shelve-based cache for persistent storage."""
    global cache
    try:
        cache = shelve.open('llm_cache', writeback=False)  # Set writeback=False for thread safety
        logger.debug("Initialized cache using shelve with writeback=False.")
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")
        logger.debug(traceback.format_exc())

def close_cache():
    """Close the shelve cache when no longer needed."""
    global cache
    with cache_lock:
        if cache is not None:
            cache.close()
            cache = None
            logger.debug("Closed the cache.")

# Register the cache close function to be called on program exit
atexit.register(close_cache)

# Function to generate a unique cache key
def generate_cache_key(user_prompt: str, history: list, summary: str, system_prompt: str, model: str) -> str:
    """
    Generate a unique cache key based on the user prompt, history, summary, system prompt, and model.
    """
    history_str = json.dumps(history)
    history_hash = hashlib.sha256(history_str.encode('utf-8')).hexdigest()
    key_components = f"{model}:{system_prompt}:{summary}:{history_hash}:{user_prompt}"
    cache_key = hashlib.sha256(key_components.encode('utf-8')).hexdigest()
    return cache_key

def remove_trailing_pass(text):
    """Remove trailing '/pass' and any trailing empty lines from the text."""
    text = text.rstrip()
    # Remove '/pass' at the end
    if text.endswith('/pass'):
        text = text[:-len('/pass')].rstrip()
    return text

def remove_pass_lines(text):
    """Remove any lines that start with '/pass' from the text."""
    lines = text.strip().splitlines()
    cleaned_lines = [line for line in lines if not line.strip().startswith('/pass')]
    return '\n'.join(cleaned_lines).strip()

def contains_pass(text):
    """Check if any line in the text starts with '/pass'."""
    lines = text.strip().splitlines()
    for line in lines:
        if line.strip().startswith('/pass'):
            return True
    return False

# Function to remove timestamps from the character's response
def remove_timestamps_from_response(response):
    """Remove timestamps at the beginning of the response."""
    # Remove timestamps like '[20-10-2024 13:45] ' at the beginning of the response
    return re.sub(r'^\[\d{2}-\d{2}-\d{4} \d{2}:\d{2}\]\s*', '', response)

# Function to remove leading character name from the response
def remove_leading_name_from_response(response, name):
    """Remove leading 'Name: ' from response if present."""
    pattern = rf'^{re.escape(name)}:\s*'
    return re.sub(pattern, '', response, count=1)

# Function to remove unwanted markdown code fences
def clean_response(response):
    """Remove unwanted markdown code fences from the response."""
    # Remove ```markdown ... ``` or ``` ... ```
    response = re.sub(r'```(?:markdown)?\s*', '', response)
    response = re.sub(r'```', '', response)
    # Remove ::: markdown ... ::: or ::: ... :::
    response = re.sub(r':::(?:\s*markdown)?\s*', '', response)
    response = re.sub(r':::', '', response)
    return response.strip()

# Function to ensure the 'characters' directory exists
def ensure_character_directory():
    """Ensure the 'characters' directory exists, create it if not"""
    if not os.path.exists(CHARACTER_DIR):
        os.makedirs(CHARACTER_DIR)
        logger.info(f"Created '{CHARACTER_DIR}' directory.")

# Function to retrieve character files
def get_character_files():
    """Retrieve a list of character YAML files from the character directory."""
    if not os.path.exists(CHARACTER_DIR):
        logger.error(f"Character directory '{CHARACTER_DIR}' does not exist.")
        return []
    character_files = [f for f in os.listdir(CHARACTER_DIR) if f.endswith('.yaml') or f.endswith('.yml')]
    logger.debug(f"Found character files: {character_files}")
    return character_files

# Function to load all character prompts
def load_all_character_prompts():
    """Load all character data from YAML files."""
    character_files = get_character_files()
    character_data = {}

    for file in character_files:
        try:
            with open(os.path.join(CHARACTER_DIR, file), 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                character_name = data.get('name', None)
                if not character_name:
                    logger.warning(f"Character file '{file}' does not contain a 'name' field.")
                    continue
                character_data[character_name] = data
                logger.debug(f"Loaded character data for '{character_name}' from '{file}'.")
        except FileNotFoundError:
            logger.error(f"Character file '{file}' not found.")
        except Exception as e:
            logger.error(f"Error loading character file '{file}': {e}")
            logger.debug(traceback.format_exc())

    return character_data

# SQLite DB functions for long-term memory storage
def init_db():
    """Initialize SQLite database for storing conversation summaries and messages."""
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY
            )
        ''')
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                name TEXT,
                content TEXT,
                timestamp TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        ''')
        # Create summaries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                summary TEXT,
                timestamp TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        ''')
        conn.commit()
    logger.info("Initialized SQLite database and ensured required tables exist.")

def create_session(session_id):
    """Create a new session in the sessions table."""
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO sessions (session_id) VALUES (?)
        ''', (session_id,))
        conn.commit()
    logger.debug(f"Created session with session_id: {session_id}")

def store_message(session_id, message):
    """Store a single message in the messages table."""
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO messages (session_id, role, name, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, message['role'], message['name'], message['content'], message['timestamp']))
        conn.commit()
    logger.debug(f"Stored message for session_id: {session_id}: {message}")

def retrieve_messages(session_id):
    """Retrieve all messages for a session_id."""
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT role, name, content, timestamp FROM messages WHERE session_id = ? ORDER BY id ASC
        ''', (session_id,))
        rows = cursor.fetchall()
    messages = []
    for row in rows:
        message = {
            'role': row[0],
            'name': row[1],
            'content': row[2],
            'timestamp': row[3]
        }
        messages.append(message)
    logger.debug(f"Retrieved messages for session_id: {session_id}: {messages}")
    return messages

def store_summary(session_id, summary_text):
    """Store a summary in the summaries table."""
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO summaries (session_id, summary, timestamp)
            VALUES (?, ?, ?)
        ''', (session_id, summary_text, timestamp))
        conn.commit()
    logger.debug(f"Stored summary for session_id: {session_id}: {summary_text}")

def retrieve_latest_summary(session_id):
    """Retrieve the latest summary for a session_id."""
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT summary FROM summaries WHERE session_id = ? ORDER BY id DESC LIMIT 1
        ''', (session_id,))
        row = cursor.fetchone()
    if row:
        summary = row[0]
        logger.debug(f"Retrieved latest summary for session_id: {session_id}: {summary}")
        return summary
    else:
        logger.debug(f"No summary found for session_id: {session_id}")
        return ""

def delete_session_data(session_id):
    """Delete session data from SQLite database."""
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM summaries WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        conn.commit()
    logger.info(f"Deleted session data for session_id: {session_id}")

def delete_messages(session_id):
    """Delete all messages for a session_id."""
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        conn.commit()
    logger.debug(f"Deleted all messages for session_id: {session_id}")

def delete_summaries(session_id):
    """Delete all summaries for a session_id."""
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM summaries WHERE session_id = ?', (session_id,))
        conn.commit()
    logger.debug(f"Deleted all summaries for session_id: {session_id}")

def retrieve_session_data(session_id):
    """Retrieve session data including history and summary from SQLite database."""
    # Retrieve messages
    history = retrieve_messages(session_id)
    # Retrieve the latest summary
    summary = retrieve_latest_summary(session_id)
    # Return
    return history, summary

# Function to add 'name' and 'timestamp' to history if missing
def add_name_to_history(history):
    """Add 'name' and 'timestamp' fields to messages in the history if missing."""
    for message in history:
        if message['role'] == 'assistant' and 'name' not in message:
            message['name'] = 'Assistant'
        if 'timestamp' not in message:
            message['timestamp'] = datetime.now().strftime("%d-%m-%Y %H:%M")
    return history

# Function to format history for Gradio Chatbot
def format_history_for_display(history):
    """
    Convert history list of dicts to list of dicts for Gradio Chatbot with type='messages'.
    Each dict should have 'role' and 'content' keys.
    """
    formatted = []
    for msg in history:
        timestamp = msg.get('timestamp', '')
        name = msg.get('name', 'Unknown')
        content = msg.get('content', '')
        role = msg.get('role', 'user')  # Ensure role is present
        if role == 'system':
            # Optionally, you can format system messages differently or skip them
            formatted.append({
                'role': 'system',
                'content': f"[{timestamp}] {content}"
            })
        else:
            formatted_content = f"[{timestamp}] **{name}**: {content}"
            formatted.append({
                'role': role,
                'content': formatted_content
            })
    logger.debug(f"Formatted history for display: {formatted}")
    return formatted

# Function to update character info display
def update_character_info():
    """Generate the content for the character info display."""
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M")
    info_text = f"**Current Date & Time:** {current_time}"
    logger.debug(f"Updated character info display: {info_text}")
    return info_text

# Function to create a new session
def create_new_session(character_prompts):
    """Create a new session ID and update the session dropdown."""
    logger.debug("Creating new session")
    # Generate a new session ID
    new_session_id = str(uuid.uuid4())
    # Create the new session in the database
    create_session(new_session_id)
    logger.debug(f"Created session with ID: {new_session_id}")
    # Update session dropdown choices
    existing_sessions = get_existing_sessions()
    logger.debug(f"Updated session list after creation: {existing_sessions}")
    # Update character info display
    character_info = update_character_info()
    # Determine assistant character
    assistant_char = "Assistant"
    logger.debug(f"Assistant character initialized to: {assistant_char}")
    # Get current user_name value or set default
    user_name_value = "User"  # Default value

    return (
        gr.update(choices=existing_sessions, value=new_session_id),  # session_id_dropdown
        gr.update(value=[]),       # chatbot
        [],                        # history state
        "",                        # summary state
        gr.update(value=character_info),                           # character_info_display
        assistant_char,                                             # assistant_character state
        gr.update(value=user_name_value)                           # user_name state
    )

# Function to retrieve existing sessions
def get_existing_sessions():
    """Retrieve a list of existing session IDs."""
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT session_id FROM sessions')
        rows = cursor.fetchall()
    session_ids = [row[0] for row in rows]
    logger.debug(f"Existing sessions: {session_ids}")
    return session_ids

# Function to delete a session
def delete_session(session_id, character_prompts):
    """Delete the selected session and update the session dropdown."""
    logger.debug(f"Deleting session ID: {session_id}")
    if session_id:
        delete_session_data(session_id)
    # Update session dropdown choices
    existing_sessions = get_existing_sessions()
    logger.debug(f"Updated session list after deletion: {existing_sessions}")

    if not existing_sessions:
        # If no sessions left, create a new one
        new_session = create_new_session(character_prompts)
        chatbot_update = new_session[1]
        history = new_session[2]
        summary = new_session[3]
        character_info_update = new_session[4]
        assistant_char = new_session[5]
        logger.debug(f"Created new session: {new_session[0]['value']}")  # Fixed line
        return (
            new_session[0],  # session_id_dropdown
            new_session[1],  # chatbot
            new_session[2],  # history
            new_session[3],  # summary
            new_session[4],  # character_info_display
            new_session[5]   # assistant_character
        )
    else:
        # If sessions exist, select the first one
        selected_session_id = existing_sessions[0]
        history, summary = retrieve_session_data(selected_session_id)
        history = add_name_to_history(history)
        formatted_history = format_history_for_display(history)
        character_info = update_character_info()

        # Determine assistant character from history
        assistant_char = None
        for msg in reversed(history):
            if msg['role'] == 'assistant':
                assistant_char = msg.get('name', None)
                break

        if not assistant_char and character_prompts:
            # If no assistant character found in history, default to the first character
            assistant_char = next(iter(character_prompts))

        logger.debug(f"Assistant character determined after deletion: {assistant_char}")

        # Prepare updates
        selected_session_update = gr.update(choices=existing_sessions, value=selected_session_id)
        chatbot_update = gr.update(value=formatted_history)
        character_info_update = gr.update(value=character_info)

        return (
            selected_session_update,  # session_id_dropdown
            chatbot_update,           # chatbot
            history,                   # history state
            summary,                   # summary state
            character_info_update,     # character_info_display
            assistant_char             # assistant_character state
        )

# Function to reset the chat
def reset_chat(session_id, character_prompts):
    """Reset the chat history and summary without deleting the session."""
    logger.debug(f"Resetting chat for session ID: {session_id}")
    # Clear the chat history and summary
    # Delete messages for the session
    delete_messages(session_id)
    logger.debug("Deleted all messages for the session.")

    # Delete summaries for the session
    delete_summaries(session_id)
    logger.debug("Deleted all summaries for the session.")

    # Update character info display
    character_info = update_character_info()
    # Determine assistant character after reset
    assistant_char = "Assistant"
    logger.debug(f"Assistant character after reset: {assistant_char}")
    # Return updated components
    logger.debug("Returning updated components after resetting the chat.")
    return (
        gr.update(value=[]),                    # chatbot
        [],                                     # history
        "",                                     # summary
        "",                                     # user_input
        gr.update(value=character_info),        # character_info_display
        assistant_char                          # assistant_character state
    )

# Queue for new messages to be processed by Gradio
new_message_queue = queue.Queue()

# Event to signal new assistant messages
new_assistant_message_event = threading.Event()

# Function to handle user input
def respond(user_input, history, summary, session_id, assistant_character, user_name):
    """Handle user input and append it to the chat history."""
    global character_prompts  # Access the global character_prompts variable
    global auto_mode_active  # Access the global auto_mode_active variable

    logger.debug(f"User input received: {user_input}")
    # Check for empty user input
    if not user_input.strip():
        logger.warning("Received empty user input.")
        return (
            gr.update(value=format_history_for_display(history)),  # chatbot
            history,                                                 # history
            summary,                                                 # summary
            "",                                                      # user_input
            gr.update(value=update_character_info()),               # character_info_display
            assistant_character                                      # assistant_character state
        )

    if not session_id:
        logger.warning("No session ID selected.")
        return (
            gr.update(value=format_history_for_display(history)),  # chatbot
            history,                                                 # history
            summary,                                                 # summary
            "",                                                      # user_input
            gr.update(value=update_character_info()),               # character_info_display
            assistant_character                                      # assistant_character state
        )

    # Get current timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

    # Ensure user_name is a string
    if isinstance(user_name, str):
        name = user_name
    elif hasattr(user_name, 'value') and isinstance(user_name.value, str):
        name = user_name.value
    else:
        name = "User"  # Fallback name

    with session_lock:
        if auto_mode_active:
            # In Auto Chat Mode: Enqueue user message to the queue
            user_message = {
                "role": "user",
                "content": user_input,
                "name": name,
                "timestamp": timestamp
            }
            history.append(user_message)
            logger.debug(f"Appended user message to history in auto mode: {user_message}")

            # Store the message in the database
            store_message(session_id, user_message)
            logger.debug("Stored user message in the database in auto mode.")

            # Enqueue the user message to the new_message_queue
            new_message_queue.put(user_message)
            logger.debug("Enqueued user message to new_message_queue.")

            # Clear the user input
            return (
                gr.update(value=format_history_for_display(history)),  # chatbot
                history,                                                 # history state
                summary,                                                 # summary state
                "",                                                      # user_input
                gr.update(value=update_character_info()),               # character_info_display
                assistant_character                                      # assistant_character state
            )
        else:
            # Normal Mode: Existing behavior of sending message to LLM
            # Append user message to history with custom name and current timestamp
            user_message = {
                "role": "user",
                "content": user_input,
                "name": name,
                "timestamp": timestamp
            }
            history.append(user_message)
            logger.debug(f"Appended user message to history: {user_message}")

            # Store the message in the database
            store_message(session_id, user_message)
            logger.debug("Stored user message in the database.")

    # Generate response from LLM outside the lock to prevent blocking
    character_name = assistant_character if assistant_character else "Assistant"
    character_data = character_prompts.get(character_name, None) if character_prompts else None

    if not character_data:
        logger.error(f"Character data for '{character_name}' not found.")
        response = "I'm sorry, I couldn't find the character data."
    else:
        response = generate_response_with_llm(user_input, history, summary, character_data, MODEL_NAME)
        if not is_first_assistant_message(history):
            # Build known_characters set
            known_characters = set()
            for msg in history:
                if 'name' in msg and msg['name']:
                    known_characters.add(msg['name'])
            if assistant_character:
                known_characters.add(assistant_character)
            # Convert to list
            known_characters_list = list(known_characters)

            # Get character description
            character_description = character_data.get('description', '')

            adjusted_response = check_and_rewrite_response(
                response,
                character_name,
                character_description,
                known_characters_list,
                history
            )
            if adjusted_response != "/pass":
                response = adjusted_response

    if response == "/skip":
        logger.info("LLM response was '/skip'. Not adding assistant message to history.")
        # Do not add to history or database, just return
        formatted_history = format_history_for_display(history)
        character_info = update_character_info()
        return (
            gr.update(value=formatted_history),      # chatbot
            history,                                 # history state
            summary,                                 # summary state
            "",                                      # user_input
            gr.update(value=character_info),         # character_info_display
            assistant_character                      # assistant_character state
        )

    if not response:
        response = "I'm sorry, I couldn't process your request."
        logger.warning("Generated empty response from LLM. Using default message.")
    else:
        response = remove_timestamps_from_response(response)
        response = remove_leading_name_from_response(response, character_name)
        response = clean_response(response)  # Clean unwanted markdown code fences
        response = remove_skip_tokens(response)
        logger.debug(f"Processed LLM response: {response}")

    # Get current timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

    # Append character message to history within the lock
    with session_lock:
        assistant_message = {
            "role": "assistant",
            "content": response,
            "name": character_name,
            "timestamp": timestamp
        }
        history.append(assistant_message)
        logger.debug(f"Appended assistant message to history: {assistant_message}")

        # Store the assistant message in the database
        store_message(session_id, assistant_message)
        logger.debug("Stored assistant message in the database.")

        # Enqueue the assistant message to the new_message_queue
        new_assistant_message_event.set()
        logger.debug("Set new_assistant_message_event.")

        # Fetch the latest history from the database
        history_fetched = retrieve_messages(session_id)
        history_fetched = add_name_to_history(history_fetched)
        logger.debug(f"After assistant response, fetched history: {history_fetched}")
        formatted_history = format_history_for_display(history_fetched)

        character_info = update_character_info()

    # Return updated components
    return (
        gr.update(value=formatted_history),      # chatbot
        history_fetched,                         # history state
        summary,                                 # summary state
        "",                                      # user_input
        gr.update(value=character_info),         # character_info_display
        assistant_character                      # assistant_character state
    )
    
# Function to start auto chat
def start_auto_chat(selected_characters, session_id):
    """Start the auto chat thread."""
    global auto_mode_active, selected_characters_global, session_id_global, auto_chat_thread

    logger.debug(f"Start Auto Chat called with characters: {selected_characters} and session_id: {session_id}")

    if not selected_characters:
        logger.warning("No characters selected for auto chat.")
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "Assistant",
            gr.update(value="User")  # Reset user_name to default if needed
        )

    if not session_id:
        logger.info("No session selected for auto chat. Creating a new session.")
        # Create a new session
        new_session_outputs = create_new_session(character_prompts)
        existing_sessions = get_existing_sessions()
        session_id = existing_sessions[-1] if existing_sessions else None
        history, summary = retrieve_session_data(session_id)
        history = add_name_to_history(history)
        logger.debug(f"Created new session: {session_id}")
    else:
        # Retrieve existing session data
        history, summary = retrieve_session_data(session_id)
        history = add_name_to_history(history)
        formatted_history = format_history_for_display(history)
        character_info = update_character_info()

        # Determine assistant character from history
        assistant_char = None
        for msg in reversed(history):
            if msg['role'] == 'assistant':
                assistant_char = msg.get('name', None)
                break

        if not assistant_char and character_prompts:
            assistant_char = next(iter(character_prompts))

    if auto_mode_active:
        logger.info("Auto chat is already active.")
        return (
            gr.update(value=format_history_for_display(history)),
            history,
            summary,
            gr.update(value=character_info),
            assistant_char,
            gr.update(value="User")  # Ensure user_name remains a string
        )

    auto_mode_active = True
    selected_characters_global = selected_characters
    session_id_global = session_id

    # Start the auto chat thread
    auto_chat_thread = threading.Thread(target=auto_chat, args=(selected_characters, session_id), daemon=True)
    auto_chat_thread.start()
    logger.info("Auto chat thread started.")

    return (
        gr.update(value=format_history_for_display(history)),
        history,
        summary,
        gr.update(value=character_info),
        assistant_char,
        gr.update(value="User")  # Ensure user_name remains a string
    )

# Function to stop auto chat
def stop_auto_chat():
    """Stop the auto chat thread."""
    global auto_mode_active
    if not auto_mode_active:
        logger.info("Auto chat is not active.")
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "Assistant",
            gr.update(value="User")  # Ensure user_name remains a string
        )
    auto_mode_active = False
    new_assistant_message_event.set()  # Unblock the generator if waiting
    logger.info("Auto chat stopped.")
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        "Assistant",
        gr.update(value="User")  # Ensure user_name remains a string
    )

# Function to summarize the conversation history
def summarize_history(history, summary, system_prompt, model, session_id, num_recent=DEFAULT_NUMBER_OF_RECENT_MESSAGES_TO_KEEP):
    """Summarize the conversation history using the LLM."""
    logger.info("Summarizing conversation history.")
    # Prepare the conversation text excluding system messages
    conversation_text = ""
    if summary:
        conversation_text += f"Previous Summary:\n{summary}\n\n"
    for msg in history:
        if msg['role'] == 'system':
            continue  # Skip system messages
        timestamp = msg.get('timestamp', '')
        name = msg.get('name', 'Unknown')
        content = msg.get('content', '')
        conversation_text += f"[{timestamp}] {name}: {content}\n"
    # Prepare the summarization prompt
    summarization_prompt = SUMMARY_PROMPT_TEMPLATE.format(conversation=conversation_text)
    logger.debug(f"Summarization prompt: {summarization_prompt}")
    # Generate cache key for summarization prompt
    cache_key_summary = generate_cache_key(summarization_prompt, history, summary, system_prompt, model)

    # Attempt to fetch summary from cache
    with cache_lock:
        if cache and cache_key_summary in cache:
            logger.debug("Cache hit for summarization prompt.")
            summary = cache[cache_key_summary]
        else:
            logger.debug("Calling LLM for summarization prompt.")
            summary = generate_response_with_llm(summarization_prompt, history, summary, system_prompt, model)

    logger.debug(f"Generated summary from LLM: {summary}")

    # Store the summary in the database
    store_summary(session_id, summary)
    logger.debug("Stored the new summary in the database.")

    return summary.strip()

# Function to generate a response from the LLM
def generate_response_with_llm(user_input: str, history: list, summary: str, character_data: dict, model: str, is_checking=False, character_description=None) -> str:
    """
    Generate a response from the LLM based on the character-specific templates and conversation context.
    Implements retry mechanism and returns '/skip' if the LLM fails after retries.
    """
    logger.debug(f"Entered generate_response_with_llm with prompt size {len(user_input)}")
    global cache

    if is_checking:
        logger.debug(f"Checking prompt with size {len(user_input)}")
        # For checking prompts, include character description in the system prompt
        if character_description:
            system_prompt = f"You are an assistant who helps to check and adjust responses according to guidelines. Use the following character description to ensure the response is in character:\n\n{character_description}"
        else:
            system_prompt = "You are an assistant who helps to check and adjust responses according to guidelines."

        # Prepare the system and user messages for the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    else:
        if not character_data:
            logger.error("Character data is required when not checking.")
            return "I'm sorry, I couldn't find the character data."

        character_name = character_data.get('name', 'Assistant')

        # Build the system prompt
        system_prompt = character_data['conversation']['system_prompt']

        # Determine if there is valid conversation history to include
        has_history = any(msg['role'] in ['user', 'assistant'] for msg in history)

        if has_history:
            # Build the conversation history string from the messages
            conversation_history = ""
            for message in history:
                if message['role'] in ['user', 'assistant']:
                    timestamp = message.get('timestamp', '')
                    name = message.get('name', 'Unknown')
                    content = message.get('content', '')
                    content = remove_skip_tokens(content)
                    conversation_history += f"[{timestamp}] {name}: {content}\n"

            # Use the character-specific user prompt template
            user_prompt_template = character_data['conversation'].get('user_prompt_template_regular', '')
            # Fill in the placeholders
            full_user_prompt = user_prompt_template.format(
                conversation_history=conversation_history,
                character_name=character_name
            )
        else:
            # Use the character-specific start prompt template
            user_prompt_template_start = character_data['conversation'].get('user_prompt_template_start', '')
            full_user_prompt = user_prompt_template_start.format(
                character_name=character_name
            )

        # Generate a unique cache key based on the current state
        cache_key = generate_cache_key(full_user_prompt, history, summary, system_prompt, model)
        logger.debug(f"Generated cache key: {cache_key}")

        # Attempt to fetch from cache
        with cache_lock:
            if cache and cache_key in cache:
                logger.info(f"Fetching result from cache for prompt: {full_user_prompt[:50]}...")
                response_content = cache[cache_key]
                logger.debug(f"Cached response: {response_content}")
                return response_content

        # Prepare the system and user messages for the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_prompt}
        ]

    # Prepare the payload
    payload = {
        "model": model,
        "messages": messages
    }

    headers = {'Content-Type': 'application/json'}

    # Implement retry mechanism
    retries = 0
    while retries < MAX_RETRIES:
        try:
            logger.info(f"Attempt {retries + 1} to send request to LLM with model '{model}'")
            logger.debug(f"Full payload:\n{json.dumps(payload, indent=2)}")

            response = requests.post(OLLAMA_URL, data=json.dumps(payload), headers=headers, stream=True, timeout=60)

            logger.debug(f"Response status code: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"Failed to generate response with LLM: HTTP {response.status_code}")
                logger.debug(f"Response content: {response.text}")
                retries += 1
                continue  # Retry

            # Process the streaming response
            response_content = ""
            raw_response = []
            for line in response.iter_lines():
                if line:
                    try:
                        line_decoded = line.decode('utf-8')
                        raw_response.append(line_decoded)  # Store the raw line for debugging
                        data = json.loads(line_decoded)
                        # Check if 'message' field is present in the data
                        if 'message' in data and 'content' in data['message']:
                            response_content += data['message']['content']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError as e:
                        logger.error(f"JSONDecodeError: {e}")
                        logger.debug(f"Line content: {line}")
                        continue

            # Check if the response content is empty
            if not response_content.strip():
                logger.warning("Received empty response from LLM.")
                logger.debug(f"Complete raw response: {raw_response}")
                retries += 1
                continue  # Retry

            # Cache the result if caching is enabled and not a checking prompt
            if cache and not is_checking:
                with cache_lock:
                    logger.debug("Caching the generated response.")
                    cache[cache_key] = response_content
                cache.sync()

            logger.debug(f"Generated response from LLM: {response_content.strip()}")
            return response_content.strip()

        except requests.exceptions.Timeout:
            logger.error("LLM API request timed out.")
            retries += 1
            continue  # Retry
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API request failed: {e}")
            logger.debug(traceback.format_exc())
            retries += 1
            continue  # Retry
        except Exception as e:
            logger.error(f"Failed to generate response with LLM: {e}")
            logger.debug(traceback.format_exc())
            retries += 1
            continue  # Retry

    # After retries, if still fails, return '/skip'
    logger.error(f"Failed to generate response after {MAX_RETRIES} retries. Returning '/skip'")
    return "/skip"
    
def check_and_rewrite_response(response: str, character_name: str, character_description: str, known_characters: List[str], history: List[dict], num_history_lines: int = DEFAULT_NUMBER_OF_HISTORY_LINES_FOR_VALIDATION) -> str:
    """
    Check and possibly rewrite the response using separate validation and correction prompts.
    Returns the adjusted response, or the original response if no adjustment is needed.
    Repeats the checking process until /pass is returned or MAX_RETRIES is reached.
    """
    # Return the response as-is if it is '/skip'
    if response.strip() == '/skip':
        return response

    known_characters_list = ', '.join(known_characters)

    # Extract the last N lines of history
    recent_history = history[-num_history_lines:] if len(history) >= num_history_lines else history[:]

    # Prepare the conversation history string
    conversation_history = ""
    for message in recent_history:
        if message['role'] in ['user', 'assistant']:
            timestamp = message.get('timestamp', '')
            name = message.get('name', 'Unknown')
            content = message.get('content', '')
            content = remove_skip_tokens(content)
            conversation_history += f"[{timestamp}] {name}: {content}\n"

    # Validation prompt template
    validation_prompt_template = """### Instructions ###
Validate the response based on the following guidelines, the character description, and the recent conversation history. Clearly indicate only /pass if it passes, or provide a concise explanation of what needs to be changed for it to pass.

### Character Description ###
{character_description}

### Recent Conversation History ###
{conversation_history}

### Guidelines ###

- **Conciseness and Creativity**
  - Respond with short, concise, and creative lines that advance the conversation.
  - Avoid stalling, looping in thought, or repetitive phrasing.

- **Character Consistency**
  - Stay fully in character, ensuring each response reflects {character_name}'s personality and the ongoing context.
  - Keep responses grounded in {character_name}’s perspective, aligning with and building upon the established conversation history without contradiction or deviation.

- **Content Relevance**
  - Include actions, feelings, dialogue, or a combination thereof, addressing the latest message or an earlier one if more relevant.
  - Advance the conversation by contributing meaningful content that enhances engagement and continuity.

- **Formatting**
  - Use *italic* formatting for inner reactions, thoughts, or subtle actions that add depth.
  - Do not use other markdown elements or mention markdown.

- **Response Structure**
  - Do not start responses with "{character_name}:" as the voice is implied.
  - If no response is required, return only "/skip".

- **Factual Consistency**
  - Ensure factual consistency with the ongoing conversation, especially regarding actions performed by characters.
  - Avoid misattributing actions or confusing character details.

- **Exclusion of Meta-commentary**
  - Exclude any meta-commentary, instructions, evaluations, or references to the chat interface or rules.
  - Responses should be solely from {character_name}'s perspective without mentioning guidelines or system instructions.
  

### Response to Validate ###
{response_to_validate}

---

### Outcome ###

- **If all guidelines are met or when deviations are minimal:** Return only /pass and nothing else. Do not provide explanation or feedback. When in doubt, return only /pass and nothing else.
- **If unmet guidelines exist and required changes are significant:** Provide only a concise explanation of required changes for compliance to only the guidelines which are not met, without specifically referencing them. Do not repeat the guidelines or mention passed validations. Do not provide suggestions for improvements outside of the scope of the guidelines. Do not introduce new criteria not explicitely mentioned before.
"""

    # Correction prompt template
    correction_prompt_template = """### Instructions ###
Based on the feedback provided, adjust the original response. Provide only the corrected response without reasoning or explanations. Do not include any feedback or guidelines in the response. If the response is too long, focus on the start of the response and ignore the rest.

---

### Original Response ###
{original_response}

### Feedback ###
{feedback}

---

### Corrected Response ###
"""

    retries = 0
    response_to_validate = response

    while retries < MAX_RETRIES:
        retries += 1

        # Prepare the validation prompt
        validation_prompt = validation_prompt_template.format(
            character_description=character_description,
            character_name=character_name,
            known_characters_list=known_characters_list,
            conversation_history=conversation_history,
            response_to_validate=response_to_validate
        )

        # Call the LLM for validation
        validation_result = generate_response_with_llm(
            user_input=validation_prompt,
            history=[],
            summary="",
            character_data=None,
            model=CHECK_MODEL_NAME,
            is_checking=True,
            character_description=character_description
        )
        #validation_result = "/pass"
        # Process the validation result
        validation_result = validation_result.strip()
        if validation_result == "/pass":
            logger.debug(f"The response passed the checks. Message: '{response_to_validate}'")
            return response_to_validate  # Return the response as it passed the check
        else:
            # Prepare the correction prompt
            correction_prompt = correction_prompt_template.format(
                original_response=response_to_validate,
                feedback=validation_result
            )

            # Call the LLM for correction
            adjusted_response = generate_response_with_llm(
                user_input=correction_prompt,
                history=[],
                summary="",
                character_data=None,
                model=CHECK_MODEL_NAME,
                is_checking=True,
                character_description=character_description
            )

            # Prepare for next iteration
            adjusted_response_cleaned = adjusted_response.strip()
            logger.debug(
                f"Attempt {retries}: The response did not pass the checks.\n"
                f"Feedback: '{validation_result}'\n"
                f"Adjusted response: '{adjusted_response_cleaned}'"
            )
            response_to_validate = adjusted_response_cleaned  # Prepare for next iteration

    logger.debug(f"Maximum retries reached. Returning last adjusted response.")
    return response_to_validate
    
# Function to handle automatic chat in background
def auto_chat(selected_characters, session_id):
    """Background thread function to handle automatic chatting."""
    global auto_mode_active, character_prompts  # Access global variables

    try:
        logger.debug(f"Auto chat started with session_id: {session_id}")

        while auto_mode_active:
            if not selected_characters:
                logger.warning("No characters selected for auto chat.")
                break

            # Iterate through characters in their selected order
            for current_character_name in selected_characters:
                if not auto_mode_active:
                    break

                current_character_data = character_prompts.get(current_character_name, None)
                if not current_character_data:
                    logger.error(f"Character data for '{current_character_name}' not found.")
                    continue
                logger.debug(f"Current character for auto chat: {current_character_name}")

                # Retrieve the character description
                character_description = current_character_data['conversation']['system_prompt']
                if not character_description:
                    logger.warning(f"Character description for '{current_character_name}' is empty.")
                    logger.warning(f"Using current_character_data: {current_character_data}")

                # Acquire the lock to safely access the history
                with session_lock:
                    history_fetched = retrieve_messages(session_id)
                    history_fetched = add_name_to_history(history_fetched)
                    logger.debug(f"Auto_chat retrieved history: {history_fetched}")

                    # Attempt to determine the assistant character from history
                    assistant_character = None
                    for msg in reversed(history_fetched):
                        if msg['role'] == 'assistant':
                            assistant_character = msg.get('name', None)
                            break

                # Generate response from LLM
                response = generate_response_with_llm(
                    user_input="",
                    history=history_fetched,
                    summary="",
                    character_data=current_character_data,
                    model=MODEL_NAME
                )
                if not is_first_assistant_message(history_fetched):
                    # Build known_characters set
                    known_characters = set()
                    for msg in history_fetched:
                        if 'name' in msg and msg['name']:
                            known_characters.add(msg['name'])
                    if selected_characters:
                        known_characters.update(selected_characters)
                    if assistant_character:
                        known_characters.add(assistant_character)
                    # Convert to list
                    known_characters_list = list(known_characters)

                    adjusted_response = check_and_rewrite_response(
                        response=response,
                        character_name=current_character_name,
                        character_description=character_description,
                        known_characters=known_characters_list,
                        history=history_fetched
                    )
                    if adjusted_response != "/pass":
                        response = adjusted_response

                if response == "/skip":
                    logger.info(f"Skipping response for {current_character_name} due to '/skip'")
                    continue  # Do not add to history or database

                if not response:
                    response = "I'm sorry, I couldn't process your request."
                    logger.warning(f"Empty response from LLM for {current_character_name}. Using default message.")
                else:
                    response = remove_timestamps_from_response(response)
                    response = remove_leading_name_from_response(response, current_character_name)
                    response = clean_response(response)
                    response = remove_skip_tokens(response)
                    logger.debug(f"Processed LLM response for {current_character_name}: {response}")

                # Get current timestamp
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

                # Prepare assistant message with correct role
                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "name": current_character_name,
                    "timestamp": timestamp
                }

                # Append the assistant message to history within the lock
                with session_lock:
                    store_message(session_id, assistant_message)
                    logger.debug("Stored assistant message in the database.")

                # Signal the frontend about the new assistant message
                new_assistant_message_event.set()

    except Exception as e:
        logger.error(f"Exception in auto_chat thread: {e}")
        logger.debug(traceback.format_exc())
        auto_mode_active = False  # Ensure the flag is reset on exception
        
script_js = """
    <script>
    document.addEventListener('keydown', function(event) {
        const activeElement = document.activeElement;
        // Ensure that the user is not typing in an input or textarea
        if (activeElement.tagName.toLowerCase() === 'input' || activeElement.tagName.toLowerCase() === 'textarea') {
            return;
        }
        if(event.key === 'ArrowLeft') {
            // Trigger previous button click
            document.getElementById('prev_button').click();
        }
        if(event.key === 'ArrowRight') {
            // Trigger next button click
            document.getElementById('next_button').click();
        }
    });

    // Auto-scroll to the latest message in the chatbot
    function scrollToBottom() {
        const chatbotElement = document.getElementById('chatbot');
        if (chatbotElement) {
            // Find the scrollable container within the chatbot
            const scrollableDiv = chatbotElement.querySelector('.overflow-auto');
            if (scrollableDiv) {
                scrollableDiv.scrollTop = scrollableDiv.scrollHeight;
            }
        }
    }

    // Observe changes in the chatbot and trigger scroll
    const chatbotObserver = new MutationObserver(function(mutations) {
        scrollToBottom();
    });

    // Start observing the chatbot container for changes
    const chatbotElement = document.getElementById('chatbot');
    if (chatbotElement) {
        const scrollableDiv = chatbotElement.querySelector('.overflow-auto');
        if (scrollableDiv) {
            chatbotObserver.observe(scrollableDiv, { childList: true, subtree: true });
        }
    }
    </script>
    """

# Function to set up the Gradio interface
def main():
    global cache, character_prompts, auto_mode_active, selected_characters_global, session_id_global, auto_chat_thread
    auto_mode_active = False
    selected_characters_global = []
    session_id_global = None
    auto_chat_thread = None

    try:
        # Initialize the SQLite DB
        init_db()
        logger.info("Starting the chatbot application.")

        # Initialize the shelve cache
        init_cache()
        logger.info("Initialized shelve cache for LLM responses.")

        # Ensure character directory exists and has default characters
        ensure_character_directory()

        # Get the list of character files
        character_files = get_character_files()

        if not character_files:
            logger.error(f"No character files found in '{CHARACTER_DIR}' directory.")
            return

        # Load the character prompts
        character_prompts = load_all_character_prompts()

        # Extract character names without '.txt'
        character_names = list(character_prompts.keys())

        with gr.Blocks(head=script_js) as demo:
            gr.Markdown("# Multi-Character Chatbot with Automatic Mode")

            # Character info display
            character_info_display = gr.Markdown()

            # Initialize session variables
            history = gr.State([])
            summary = gr.State("")
            assistant_character = gr.State(None)  # State for assistant character
            user_name = gr.State("User")  # State for user name

            # Wrap chat interface in a container for visibility management
            with gr.Row() as chat_container:
                with gr.Column():
                    with gr.Row():
                        session_id_dropdown = gr.Dropdown(
                            choices=get_existing_sessions(),
                            label="Select Session",
                            value=None,
                            interactive=True
                        )
                        new_session_button = gr.Button("Create New Session")
                        delete_session_button = gr.Button("Delete Session")
                        slideshow_button = gr.Button("Slideshow")  # Added Slideshow Button

                    # User name input and assistant character selection
                    with gr.Row():
                        user_name_input = gr.Textbox(
                            label="Your Name",
                            value="User",
                            placeholder="Enter your name",
                            interactive=True
                        )
                        assistant_character_dropdown = gr.Dropdown(
                            choices=character_names,
                            label="Select Assistant Character",
                            value=None,
                            interactive=True
                        )

                    # Update user_name state when user_name_input changes
                    user_name_input.change(
                        lambda x: x.strip() if x.strip() else "User",
                        inputs=user_name_input,
                        outputs=user_name
                    )

                    # Auto chat controls
                    with gr.Group():
                        gr.Markdown("### Auto Chat Controls")
                        with gr.Row():
                            selected_characters = gr.CheckboxGroup(
                                choices=character_names,
                                label="Select Characters for Auto Chat (Order determines speaking order)",
                                value=[]
                            )
                            participant_order = gr.Markdown("Current speaking order: None")
                        with gr.Row():
                            start_auto_button = gr.Button("Start Auto Chat")
                            stop_auto_button = gr.Button("Stop Auto Chat")

                    # Function to update participant order display
                    def update_participant_order(selected):
                        if not selected:
                            return "Current speaking order: None"
                        order_text = "Current speaking order: " + " → ".join(selected)
                        return order_text

                    # Update participant order when selection changes
                    selected_characters.change(
                        update_participant_order,
                        inputs=[selected_characters],
                        outputs=[participant_order]
                    )

                    # Initialize Chatbot with 'messages' type
                    chatbot = gr.Chatbot(type='messages', elem_id='chatbot')
                    user_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here and press Enter",
                        lines=1
                    )

                    # Reset chat button
                    reset_button = gr.Button("Reset Chat")

            # Slideshow Interface Components (Initially Hidden)
            with gr.Row(visible=False, elem_id="slideshow_container") as slideshow_container:
                slideshow_text = gr.Markdown("**Slideshow Display**", visible=True, elem_id="slideshow_text")
                with gr.Row():
                    first_button = gr.Button("⏮️ First", elem_id="first_button")  # Added First Button
                    prev_button = gr.Button("◀️ Previous", elem_id="prev_button")
                    next_button = gr.Button("Next ▶️", elem_id="next_button")
                    last_button = gr.Button("Last ⏭️", elem_id="last_button")    # Added Last Button
                    back_button = gr.Button("Back to Chat")

            # Slideshow State
            slideshow_index = gr.State(0)

            # Event: Load existing sessions and set default session on app load
            def load_default_session():
                logger.debug("Loading default session on app load.")
                try:
                    existing_sessions = get_existing_sessions()
                    logger.debug(f"Existing sessions on load: {existing_sessions}")

                    if existing_sessions:
                        selected_session_id = existing_sessions[0]  # Select the first session
                        history_value, summary_value = retrieve_session_data(selected_session_id)
                    else:
                        # If no existing sessions, create a new one
                        new_session_outputs = create_new_session(character_prompts)
                        existing_sessions = get_existing_sessions()
                        selected_session_id = existing_sessions[-1] if existing_sessions else None
                        history_value, summary_value = retrieve_session_data(selected_session_id)
                        logger.debug(f"Created new session on load: {selected_session_id}")

                    # Add further logic to handle history and assistant_char
                    history_value = add_name_to_history(history_value)
                    formatted_history = format_history_for_display(history_value)
                    character_info = update_character_info()

                    # Determine assistant character from history
                    assistant_char = None
                    for msg in reversed(history_value):
                        if msg['role'] == 'assistant':
                            assistant_char = msg.get('name', None)
                            break

                    if not assistant_char and character_prompts:
                        assistant_char = next(iter(character_prompts))

                    logger.debug(f"Assistant character on load: {assistant_char}")

                    # Get current user_name value or set default
                    user_name_value = "User"  # Default value
                    # Ensure 'user_name' is a Gradio State object and extract its value
                    if isinstance(user_name, gr.State):
                        if user_name.value and isinstance(user_name.value, str):
                            user_name_value = user_name.value
                    elif isinstance(user_name, str):
                        user_name_value = user_name

                    return (
                        gr.update(choices=existing_sessions, value=selected_session_id),  # session_id_dropdown
                        gr.update(value=formatted_history),  # chatbot
                        history_value,                        # history state
                        summary_value,                        # summary state
                        gr.update(value=character_info),      # character_info_display
                        assistant_char,                       # assistant_character state
                        gr.update(value=user_name_value),     # user_name state
                        gr.update(visible=True),              # Show chat_container
                        gr.update(visible=False),             # Hide slideshow_container
                        len(history_value) - 1 if history_value else 0  # Initialize slideshow_index to last message
                    )
                except Exception as e:
                    logger.error(f"Error loading default session: {e}")
                    logger.debug(traceback.format_exc())
                    # Return default safe values to prevent Gradio from crashing
                    return (
                        gr.update(choices=[], value=None),
                        gr.update(value=[]),
                        [],
                        "",
                        gr.update(value="**Current Date & Time:** N/A"),
                        "Assistant",
                        gr.update(value="User"),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        0  # Reset slideshow_index
                    )

            demo.load(
                load_default_session,
                inputs=None,
                outputs=[
                    session_id_dropdown,
                    chatbot,
                    history,
                    summary,
                    character_info_display,
                    assistant_character,
                    user_name,
                    chat_container,
                    slideshow_container,
                    slideshow_index
                ]
            )

            # Event: Session ID change
            def on_session_change(session_id_value):
                logger.debug(f"Session ID changed to: {session_id_value}")
                try:
                    if not session_id_value:
                        logger.info("Session ID is None. Resetting components.")
                        return (
                            gr.update(value=format_history_for_display([])),  # chatbot
                            [],                                                 # history
                            "",                                                 # summary
                            gr.update(value=update_character_info()),         # character_info_display
                            "Assistant",                                        # assistant_character state
                            gr.update(visible=True),                           # Show chat_container
                            gr.update(visible=False),                          # Hide slideshow_container
                            0  # Reset slideshow_index
                        )
                    with session_lock:
                        history_value, summary_value = retrieve_session_data(session_id_value)
                        history_value = add_name_to_history(history_value)
                        formatted_history = format_history_for_display(history_value)
                        character_info = update_character_info()

                        # Attempt to determine the assistant character from history
                        assistant_char = None
                        for msg in reversed(history_value):
                            if msg['role'] == 'assistant':
                                assistant_char = msg.get('name', None)
                                break

                        if not assistant_char and character_prompts:
                            assistant_char = next(iter(character_prompts))

                        logger.debug(f"Assistant character determined: {assistant_char}")

                    # Get current user_name value or set default
                    user_name_value = "User"  # Default value
                    if isinstance(user_name, gr.State):
                        if user_name.value and isinstance(user_name.value, str):
                            user_name_value = user_name.value
                    elif isinstance(user_name, str):
                        user_name_value = user_name

                    return (
                        gr.update(value=formatted_history),
                        history_value,
                        summary_value,
                        gr.update(value=character_info),
                        assistant_char,          # Update assistant_character state
                        gr.update(visible=True),  # Show chat_container
                        gr.update(visible=False),  # Hide slideshow_container
                        len(history_value) -1 if history_value else 0  # Set slideshow_index to last message
                    )
                except Exception as e:
                    logger.error(f"Error in on_session_change: {e}")
                    logger.debug(traceback.format_exc())
                    # Return default safe values to prevent Gradio from crashing
                    return (
                        gr.update(value=[]),
                        [],
                        "",
                        gr.update(value="**Current Date & Time:** N/A"),
                        "Assistant",
                        gr.update(visible=True),
                        gr.update(visible=False),
                        0  # Reset slideshow_index
                    )

            session_id_dropdown.change(
                on_session_change,
                inputs=session_id_dropdown,
                outputs=[
                    chatbot,
                    history,
                    summary,
                    character_info_display,
                    assistant_character,
                    chat_container,
                    slideshow_container,
                    slideshow_index
                ]
            )

            # Event: Handle user input submission
            user_input.submit(
                respond,
                inputs=[user_input, history, summary, session_id_dropdown, assistant_character, user_name],
                outputs=[chatbot, history, summary, user_input, character_info_display, assistant_character]
            )

            # Event: Create new session
            def handle_create_new_session():
                """Handle the creation of a new session."""
                logger.debug("Handling creation of a new session.")
                try:
                    return create_new_session(character_prompts)
                except Exception as e:
                    logger.error(f"Error in handle_create_new_session: {e}")
                    logger.debug(traceback.format_exc())
                    # Return default safe values to prevent Gradio from crashing
                    return (
                        gr.update(choices=[], value=None),
                        gr.update(value=[]),
                        [],
                        "",
                        gr.update(value="**Current Date & Time:** N/A"),
                        "Assistant",
                        gr.update(value="User")
                    )

            new_session_button.click(
                handle_create_new_session,
                inputs=None,
                outputs=[session_id_dropdown, chatbot, history, summary, character_info_display, assistant_character, user_name]
            )

            # Event: Delete session
            def handle_delete_session(session_id_value):
                logger.debug(f"Deleting session ID: {session_id_value}")
                if session_id_value:
                    delete_session_data(session_id_value)
                    logger.info(f"Deleted session data for session_id: {session_id_value}")
                # Update session dropdown choices
                existing_sessions = get_existing_sessions()
                logger.debug(f"Updated session list after deletion: {existing_sessions}")

                if not existing_sessions:
                    # If no sessions left, create a new one
                    new_session_outputs = create_new_session(character_prompts)
                    return (
                        new_session_outputs[0],  # session_id_dropdown
                        new_session_outputs[1],  # chatbot
                        new_session_outputs[2],  # history
                        new_session_outputs[3],  # summary
                        new_session_outputs[4],  # character_info_display
                        new_session_outputs[5],  # assistant_character
                        new_session_outputs[6]   # user_name
                    )
                else:
                    # If sessions exist, select the first one
                    selected_session_id = existing_sessions[0]
                    history, summary = retrieve_session_data(selected_session_id)
                    history = add_name_to_history(history)
                    formatted_history = format_history_for_display(history)
                    character_info = update_character_info()

                    # Determine assistant character from history
                    assistant_char = None
                    for msg in reversed(history):
                        if msg['role'] == 'assistant':
                            assistant_char = msg.get('name', None)
                            break

                    if not assistant_char and character_prompts:
                        assistant_char = next(iter(character_prompts))

                    logger.debug(f"Assistant character determined after deletion: {assistant_char}")

                    # Get current user_name value or set default
                    user_name_value = "User"  # Default value
                    if isinstance(user_name, gr.State):
                        if user_name.value and isinstance(user_name.value, str):
                            user_name_value = user_name.value
                    elif isinstance(user_name, str):
                        user_name_value = user_name

                    return (
                        gr.update(choices=existing_sessions, value=selected_session_id),  # session_id_dropdown
                        gr.update(value=formatted_history),  # chatbot
                        history,                             # history state
                        summary,                             # summary state
                        gr.update(value=character_info),     # character_info_display
                        assistant_char,                      # assistant_character state
                        gr.update(value=user_name_value)     # user_name state
                    )

            delete_session_button.click(
                handle_delete_session,
                inputs=session_id_dropdown,
                outputs=[session_id_dropdown, chatbot, history, summary, character_info_display, assistant_character, user_name]
            )

            # Event: Reset chat
            def handle_reset_chat(session_id_value):
                return reset_chat(session_id_value, character_prompts)

            reset_button.click(
                handle_reset_chat,
                inputs=session_id_dropdown,
                outputs=[chatbot, history, summary, user_input, character_info_display, assistant_character]
            )

            # Event: Start auto chat
             # Event: Start auto chat with ordered participants
            def handle_start_auto_chat(selected_characters_value, session_id_value):
                """Generator function for auto chat, streaming new messages to the front-end."""
                global character_prompts

                logger.debug(f"Starting auto-chat for session_id: {session_id_value} with ordered characters: {selected_characters_value}")

                if not auto_mode_active:
                    # Pass the characters in their selected order
                    start_auto_chat(selected_characters_value, session_id_value)

                # Initial fetch of history and summary
                with session_lock:
                    history_fetched, summary_fetched = retrieve_session_data(session_id_value)
                    history_fetched = add_name_to_history(history_fetched)
                    formatted_history = format_history_for_display(history_fetched)
                    character_info = update_character_info()

                    # Determine assistant character from history
                    assistant_char = None
                    for msg in reversed(history_fetched):
                        if msg['role'] == 'assistant':
                            assistant_char = msg.get('name', None)
                            break

                    if not assistant_char and character_prompts:
                        assistant_char = next(iter(character_prompts))

                # Initial yield
                yield (
                    gr.update(value=formatted_history),
                    history_fetched,
                    summary_fetched,
                    "",
                    gr.update(value=character_info),
                    assistant_char
                )

                # Continuous updates
                while auto_mode_active:
                    event_set = new_assistant_message_event.wait(timeout=1)
                    if event_set:
                        with session_lock:
                            history_fetched, summary_fetched = retrieve_session_data(session_id_value)
                            history_fetched = add_name_to_history(history_fetched)
                            formatted_history = format_history_for_display(history_fetched)
                            character_info = update_character_info()

                            assistant_char = None
                            for msg in reversed(history_fetched):
                                if msg['role'] == 'assistant':
                                    assistant_char = msg.get('name', None)
                                    break

                            if not assistant_char and character_prompts:
                                assistant_char = next(iter(character_prompts))

                        yield (
                            gr.update(value=formatted_history),
                            history_fetched,
                            summary_fetched,
                            "",
                            gr.update(value=character_info),
                            assistant_char
                        )

                        new_assistant_message_event.clear()
                    else:
                        continue

            # Connect start button to handler
            start_auto_button.click(
                handle_start_auto_chat,
                inputs=[selected_characters, session_id_dropdown],
                outputs=[chatbot, history, summary, user_input, character_info_display, assistant_character],
                queue=True
            )

            # Event: Start auto chat
            start_auto_button.click(
                handle_start_auto_chat,
                inputs=[selected_characters, session_id_dropdown],
                outputs=[chatbot, history, summary, user_input, character_info_display, assistant_character],
                queue=True
            )

            # Event: Stop auto chat
            def handle_stop_auto_chat():
                """Stop the auto chat thread."""
                global auto_mode_active
                if not auto_mode_active:
                    logger.info("Auto chat is not active.")
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        "Assistant",
                        gr.update(value="User")  # Ensure user_name remains a string
                    )
                auto_mode_active = False
                logger.info("Auto chat stopped.")
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "Assistant",
                    gr.update(value="User")  # Ensure user_name remains a string
                )

            stop_auto_button.click(
                handle_stop_auto_chat,
                inputs=None,
                outputs=[chatbot, history, summary, user_input, character_info_display, assistant_character]
            )

            # Assistant character selection
            def on_assistant_character_change(assistant_char_value):
                logger.debug(f"Assistant character changed to: {assistant_char_value}")
                return assistant_char_value

            assistant_character_dropdown.change(
                on_assistant_character_change,
                inputs=assistant_character_dropdown,
                outputs=assistant_character
            )

                        # Slideshow Functions
            def enter_slideshow(session_id_value):
                """Function to enter slideshow mode."""
                logger.debug(f"Entering slideshow mode with session_id: {session_id_value}")
                try:
                    if not session_id_value:
                        logger.warning("No session selected for slideshow.")
                        return (
                            gr.update(visible=True),
                            gr.update(visible=False),
                            0
                        )
                    with session_lock:
                        history_value, _ = retrieve_session_data(session_id_value)
                        if not history_value:
                            logger.warning("No history available for slideshow.")
                            return (
                                gr.update(visible=True),
                                gr.update(visible=False),
                                0
                            )
                        # Initialize slideshow_index to the last message
                        initial_index = len(history_value) - 1 if history_value else 0
                        # Update slideshow_text with the current message
                        message = history_value[initial_index]
                        display_text = f"""**{message['name']}**  
_{message['timestamp']}_

{message['content']}"""
                        slideshow_text.value = display_text
                    return (
                        gr.update(visible=False),  # Hide chat_container
                        gr.update(visible=True),   # Show slideshow_container
                        initial_index               # Initialize slideshow_index to last message
                    )
                except Exception as e:
                    logger.error(f"Error entering slideshow: {e}")
                    logger.debug(traceback.format_exc())
                    return gr.update(visible=True), gr.update(visible=False), 0

            def update_slideshow(index, history):
                """Update the slideshow display based on the current index."""
                logger.debug(f"Updating slideshow at index: {index}")
                try:
                    if not history:
                        logger.warning("No messages to display in slideshow.")
                        return "No messages to display.", index
                    # Adjust index if out of bounds
                    index = max(0, min(index, len(history) - 1))
                    message = history[index]
                    display_text = f"""**{message['name']}**  
_{message['timestamp']}_

{message['content']}"""
                    logger.debug(f"Displaying message: {display_text}")
                    return display_text, index
                except Exception as e:
                    logger.error(f"Error updating slideshow: {e}")
                    logger.debug(traceback.format_exc())
                    return "Error displaying message.", index

            def prev_slideshow(index, history):
                """Go to the previous message in the slideshow."""
                logger.debug("Navigating to previous message in slideshow.")
                if index > 0:
                    index -= 1
                return update_slideshow(index, history)

            def next_slideshow(index, history):
                """Go to the next message in the slideshow."""
                logger.debug("Navigating to next message in slideshow.")
                if index < len(history) - 1:
                    index += 1
                return update_slideshow(index, history)

            def first_slideshow(history):
                """Go to the first message in the slideshow."""
                logger.debug("Navigating to first message in slideshow.")
                return update_slideshow(0, history)

            def last_slideshow(history):
                """Go to the last message in the slideshow."""
                logger.debug("Navigating to last message in slideshow.")
                last_index = len(history) - 1 if history else 0
                return update_slideshow(last_index, history)

            def exit_slideshow():
                """Exit slideshow mode and return to chat interface."""
                logger.debug("Exiting slideshow mode.")
                return (
                    gr.update(visible=True),   # Show chat_container
                    gr.update(visible=False),  # Hide slideshow_container
                )

            # Slideshow Button Click
            slideshow_button.click(
                enter_slideshow,
                inputs=session_id_dropdown,  # Pass session_id_dropdown's value directly
                outputs=[chat_container, slideshow_container, slideshow_index]
            )

            # Previous Button Click
            prev_button.click(
                prev_slideshow,
                inputs=[slideshow_index, history],
                outputs=[slideshow_text, slideshow_index]
            )

            # Next Button Click
            next_button.click(
                next_slideshow,
                inputs=[slideshow_index, history],
                outputs=[slideshow_text, slideshow_index]
            )

            # First Button Click
            first_button.click(
                first_slideshow,
                inputs=[history],
                outputs=[slideshow_text, slideshow_index]
            )

            # Last Button Click
            last_button.click(
                last_slideshow,
                inputs=[history],
                outputs=[slideshow_text, slideshow_index]
            )

            # Back Button Click
            back_button.click(
                exit_slideshow,
                inputs=None,
                outputs=[chat_container, slideshow_container]
            )

            # Slideshow Display Update on Slideshow Index Change
            slideshow_index.change(
                update_slideshow,
                inputs=[slideshow_index, history],
                outputs=[slideshow_text, slideshow_index]
            )

        logger.info("Launching Gradio app.")
        # Launch Gradio with share=False for local access
        try:
            demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
        except Exception as e:
            logger.error(f"Failed to launch Gradio app: {e}")
            logger.debug(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error initializing the main function: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.debug(traceback.format_exc())
