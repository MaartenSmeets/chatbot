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

# Define constants
OLLAMA_URL = "http://localhost:11434/api/chat"  # Replace with your Ollama endpoint if different
MODEL_NAME = "vanilj/midnight-miqu-70b-v1.5:latest"  # The model version
CHARACTER_DIR = 'characters'  # Directory where character text files are stored
LOG_FILE = 'app.log'  # Log file path

# Summarization settings (configurable)
MAX_CONTEXT_LENGTH = 100000  # Max context length before summarizing
DEFAULT_NUMBER_OF_RECENT_MESSAGES_TO_KEEP = 8  # Configurable number of recent messages to keep in history after summarizing
SUMMARY_PROMPT_TEMPLATE = (
    "Please provide a concise but comprehensive summary of the following conversation, "
    "including all important details, timestamps, and topics discussed:\n{conversation}\nSummary:"
)

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)

# Create formatter and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Global cache variables
cache = None
cache_lock = threading.RLock()  # Use RLock for reentrant locking

# Message queue for auto chat
message_queue = queue.Queue()

# Global state to control auto mode
auto_mode_active = False
selected_characters_global = []
session_id_global = None
auto_chat_thread = None

# Shared history list for auto updates
shared_history = []
shared_history_lock = threading.Lock()

# Queue for new messages to be processed by Gradio
new_message_queue = queue.Queue()

# Ensure the 'characters' directory exists
def ensure_character_directory():
    """Ensure the 'characters' directory exists, create it if not, and add default characters."""
    if not os.path.exists(CHARACTER_DIR):
        os.makedirs(CHARACTER_DIR)
        logger.info(f"Created '{CHARACTER_DIR}' directory.")
        # Add default character files if necessary
        default_characters = {
            'Luna.txt': "You are Luna, a compassionate and knowledgeable life coach with a warm, coaching personality. You have years of experience in emotional wellness, personal development, and psychology, and you use this expertise to guide others toward becoming the best version of themselves.",
            'Orion.txt': "You are Orion, a thoughtful and analytical man with a calm, intellectual presence. With a tall, lean frame and sharp features, you have an air of quiet contemplation and curiosity that draws people in.",
            'Solara.txt': "You are Solara, a warm and engaging woman known for your natural beauty, charm, and deep emotional intelligence. With a petite, graceful figure, radiant skin, and long, flowing hair, you effortlessly draw people in with your expressive eyes and captivating presence."
        }
        for filename, content in default_characters.items():
            file_path = os.path.join(CHARACTER_DIR, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Added default character file: {file_path}")

# Function to retrieve character files
def get_character_files():
    """Retrieve a list of character files from the character directory."""
    if not os.path.exists(CHARACTER_DIR):
        logger.error(f"Character directory '{CHARACTER_DIR}' does not exist.")
        return []
    character_files = [f for f in os.listdir(CHARACTER_DIR) if f.endswith('.txt')]
    logger.debug(f"Found character files: {character_files}")
    return character_files

# SQLite DB functions for long-term memory storage
def init_db():
    """Initialize SQLite database for storing conversation summaries and histories."""
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                history TEXT,
                summary TEXT
            )
        ''')
        conn.commit()
    logger.info("Initialized SQLite database and ensured 'sessions' table exists.")

def store_session_data(session_id, history, summary):
    """Store session data including history and summary in SQLite database."""
    history_json = json.dumps(history)
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            REPLACE INTO sessions (session_id, history, summary)
            VALUES (?, ?, ?)
        ''', (session_id, history_json, summary))
        conn.commit()
    logger.debug(f"Stored session data for session_id: {session_id}")

def retrieve_session_data(session_id):
    """Retrieve session data including history and summary from SQLite database."""
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT history, summary FROM sessions WHERE session_id = ?
        ''', (session_id,))
        row = cursor.fetchone()
    if row:
        history_json, summary = row
        history = json.loads(history_json) if history_json else []
        logger.debug(f"Retrieved session data for session_id: {session_id}")
        return history, summary
    logger.debug(f"No session data found for session_id: {session_id}")
    return [], ""

def delete_session_data(session_id):
    """Delete session data from SQLite database."""
    with sqlite3.connect('memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        conn.commit()
    logger.info(f"Deleted session data for session_id: {session_id}")

# Cache initialization using shelve
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

# Unified System Prompt Including All Characters
def load_all_character_prompts():
    """Load all character prompts and combine them into a single system prompt."""
    character_files = get_character_files()
    combined_prompts = ""
    character_prompts = {}

    for file in character_files:
        try:
            with open(os.path.join(CHARACTER_DIR, file), 'r', encoding='utf-8') as f:
                prompt = f.read()
                character_name = get_character_name(file)
                character_prompts[character_name] = prompt.strip()
                logger.debug(f"Loaded system prompt for '{character_name}' from '{file}'.")
        except FileNotFoundError:
            logger.error(f"Character file '{file}' not found.")
        except Exception as e:
            logger.error(f"Error loading character file '{file}': {e}")
            logger.debug(traceback.format_exc())

    return character_prompts

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

# Function to extract character name from file or prompt
def get_character_name(character_file_or_prompt):
    """Extract character name from character file name or prompt."""
    if isinstance(character_file_or_prompt, str):
        if character_file_or_prompt.endswith('.txt'):
            return os.path.splitext(character_file_or_prompt)[0]
        else:
            # If it's a prompt, try to extract the name from "You are [Name]" or "[Name], ..." pattern
            match = re.search(r"You are (\w+)", character_file_or_prompt)
            if not match:
                match = re.search(r"^(\w+),\s*", character_file_or_prompt)
            if match:
                return match.group(1)
    return "Assistant"  # Default name if extraction fails

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
    return gr.update(value=info_text)

# Function to create a new session
def create_new_session():
    """Create a new session ID and update the session dropdown."""
    logger.debug(f"Creating new session")
    # Generate a new session ID
    new_session_id = str(uuid.uuid4())
    # Initialize session data
    history = []
    summary = ""
    # Store initial session data
    store_session_data(new_session_id, history, summary)
    logger.debug(f"Stored session data for new session ID: {new_session_id}")
    # Update session dropdown choices
    existing_sessions = get_existing_sessions()
    logger.debug(f"Updated session list after creation: {existing_sessions}")
    # Update character info display
    character_info = update_character_info()
    # Determine assistant character
    assistant_char = "Assistant"
    logger.debug(f"Assistant character initialized to: {assistant_char}")
    # Return the updated session dropdown and select the new session
    logger.debug("Returning updated components after creating a new session.")
    return (
        gr.update(choices=existing_sessions, value=new_session_id),  # session_id_dropdown
        gr.update(value=format_history_for_display(history)),        # chatbot
        history,                                                   # history
        summary,                                                   # summary
        character_info,                                           # character_info_display
        assistant_char                                            # assistant_character state
    )

# Function to delete a session
def delete_session(session_id):
    """Delete the selected session and update the session dropdown."""
    logger.debug(f"Deleting session ID: {session_id}")
    if session_id:
        delete_session_data(session_id)
    # Update session dropdown choices
    existing_sessions = get_existing_sessions()
    logger.debug(f"Updated session list after deletion: {existing_sessions}")

    if not existing_sessions:
        # If no sessions left, create a new one
        selected_session_id = str(uuid.uuid4())
        history = []
        summary = ""
        store_session_data(selected_session_id, history, summary)
        existing_sessions = [selected_session_id]
        logger.debug(f"Created new session: {selected_session_id}")
    else:
        # If sessions exist, select the first one
        selected_session_id = existing_sessions[0]
        history, summary = retrieve_session_data(selected_session_id)
        history = add_name_to_history(history)

    # Update character info display
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

    # Return the updated session dropdown and related components, including session_id state
    logger.debug("Returning updated components after deleting a session.")
    return (
        gr.update(choices=existing_sessions, value=selected_session_id),  # session_id_dropdown
        gr.update(value=format_history_for_display(history)),            # chatbot
        history,                                                        # history
        summary,                                                        # summary
        character_info,                                                # character_info_display
        assistant_char                                                 # assistant_character state
    )

# Function to reset the chat
def reset_chat(session_id):
    """Reset the chat history and summary without deleting the session."""
    logger.debug(f"Resetting chat for session ID: {session_id}")
    # Clear the chat history and summary
    history = []
    summary = ""
    # Update the session data to reset history and summary
    if session_id:
        store_session_data(session_id, history, summary)
        logger.debug(f"Session data reset for session ID: {session_id}")
    # Update character info display
    character_info = update_character_info()
    # Determine assistant character after reset
    assistant_char = "Assistant"
    logger.debug(f"Assistant character after reset: {assistant_char}")
    # Return updated components
    logger.debug("Returning updated components after resetting the chat.")
    return (
        gr.update(value=format_history_for_display(history)),  # chatbot
        history,                                           # history
        summary,                                           # summary
        gr.update(value=''),                              # user_input
        character_info,                                   # character_info_display
        assistant_char                                    # assistant_character state
    )

# Function to handle user input
def respond(user_input, history, summary, session_id, assistant_character):
    """Handle user input and append it to the chat history."""
    logger.debug(f"User input received: {user_input}")
    # Check for empty user input
    if not user_input.strip():
        logger.warning("Received empty user input.")
        return (
            gr.update(value=format_history_for_display(history)),  # chatbot
            history,                                               # history
            summary,                                               # summary
            gr.update(value=''),                                  # user_input
            session_id,                                            # session_id state
            assistant_character                                    # assistant_character state
        )

    if not session_id:
        logger.warning("No session ID selected.")
        return (
            gr.update(value=format_history_for_display(history)),  # chatbot
            history,                                               # history
            summary,                                               # summary
            gr.update(value=''),                                  # user_input
            session_id,                                            # session_id state
            assistant_character                                    # assistant_character state
        )

    # Append user message to history with 'User' as the name and current timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")
    history.append({
        "role": "user",
        "content": user_input,
        "name": "User",
        "timestamp": timestamp
    })
    logger.debug(f"Appended user message to history: {history[-1]}")

    # Store the updated session data
    store_session_data(session_id, history, summary)
    logger.debug("Updated session data stored.")

    # Generate response from LLM
    character_prompts = load_all_character_prompts()
    character_name = "Assistant"  # Default name if not found; adjust as needed
    character_prompt = ""

    if assistant_character:
        character_name = get_character_name(assistant_character)
        character_prompt = character_prompts.get(character_name, "")
    elif character_prompts:
        # If no assistant character is selected, default to the first character
        character_name = next(iter(character_prompts))
        character_prompt = character_prompts[character_name]
    else:
        logger.warning("No character prompts available. Using default 'Assistant'.")
        character_prompt = ""

    logger.debug(f"Assistant Character: {character_name}, Prompt Length: {len(character_prompt)}")

    response = generate_response_with_llm(user_input, history, summary, character_prompt, MODEL_NAME)

    if not response:
        response = "I'm sorry, I couldn't process your request."
        logger.warning("Generated empty response from LLM. Using default message.")
    else:
        response = remove_timestamps_from_response(response)
        response = remove_leading_name_from_response(response, character_name)
        response = clean_response(response)  # Clean unwanted markdown code fences
        logger.debug(f"Processed LLM response: {response}")

    # Append character message to history
    assistant_message = {
        "role": "assistant",
        "content": response,
        "name": character_name,
        "timestamp": timestamp  # Use original timestamp or generate new one
    }
    history.append(assistant_message)
    logger.debug(f"Appended assistant message to history: {assistant_message}")

    # Store the updated session data with the character's response
    store_session_data(session_id, history, summary)
    logger.debug("Character response stored in session data.")

    # Update shared history for automatic updates
    with shared_history_lock:
        shared_history.append(assistant_message)
        logger.debug(f"Shared history updated with new message: {assistant_message}")

    # Add new message to the queue for the front-end to process
    new_message_queue.put(assistant_message)

    return (
        gr.update(value=format_history_for_display(history)),  # chatbot
        history,                                               # history
        summary,                                               # summary
        gr.update(value=''),                                  # user_input
        session_id,                                            # session_id state
        assistant_character                                    # assistant_character state
    )

# Function to start auto chat
def start_auto_chat(selected_characters, session_id, auto_refresh_trigger):
    """Start the auto chat thread."""
    global auto_mode_active, selected_characters_global, session_id_global, auto_chat_thread, cache, character_prompts

    logger.debug(f"Start Auto Chat called with characters: {selected_characters} and session_id: {session_id}")

    if not selected_characters:
        logger.warning("No characters selected for auto chat.")
        # Return without starting auto chat
        return (
            gr.update(),  # chatbot remains unchanged
            history_placeholder(),
            summary_placeholder(),
            gr.update(),  # character_info_display remains unchanged
            auto_refresh_trigger  # No change
        )

    if not session_id:
        logger.info("No session selected for auto chat. Creating a new session.")
        # Create a new session and update the session_id
        updated_components = create_new_session()
        # Unpack the returned components
        session_id_dropdown_update, chatbot_update, history, summary, character_info_display, assistant_char = updated_components
        logger.debug(f"Created and selected new session ID: {session_id_dropdown_update.value}")
        # Update the session_id_global with the new session_id
        session_id = session_id_dropdown_update.value

    if auto_mode_active:
        logger.info("Auto chat is already active.")
        return (
            gr.update(),  # chatbot remains unchanged
            history_placeholder(),
            summary_placeholder(),
            gr.update(),  # character_info_display remains unchanged
            auto_refresh_trigger  # No change
        )

    auto_mode_active = True
    selected_characters_global = selected_characters
    session_id_global = session_id

    # Make sure character prompts are loaded
    character_prompts = load_all_character_prompts()

    # Start the auto chat thread
    auto_chat_thread = threading.Thread(target=auto_chat, daemon=True)
    auto_chat_thread.start()
    logger.info("Auto chat thread started.")

    # Trigger an immediate refresh by incrementing the trigger
    auto_refresh_trigger += 1

    logger.debug("Auto refresh triggered upon starting auto chat.")

    return (
        gr.update(),  # chatbot remains unchanged
        history_placeholder(),
        summary_placeholder(),
        gr.update(),  # character_info_display remains unchanged
        auto_refresh_trigger  # Update the trigger
    )

def history_placeholder():
    """Return the current history for placeholders."""
    return None  # Adjust based on your UI needs

def summary_placeholder():
    """Return the current summary for placeholders."""
    return None  # Adjust based on your UI needs

def auto_chat():
    """Background thread function to handle automatic chatting."""
    global auto_mode_active, selected_characters_global, session_id_global, cache, character_prompts
    current_index = 0
    try:
        # Retrieve history and summary at the start
        history, summary = retrieve_session_data(session_id_global)
        history = add_name_to_history(history)
        logger.debug(f"Auto chat started with session_id: {session_id_global}")

        while auto_mode_active:
            if not selected_characters_global:
                logger.warning("No characters selected for auto chat.")
                break

            current_character_file = selected_characters_global[current_index]
            current_character_name = get_character_name(current_character_file)
            current_character_prompt = character_prompts.get(current_character_name, "")
            logger.debug(f"Current character for auto chat: {current_character_name}")

            # Prepare user_prompt from the last message or default
            if history and history[-1]['role'] == 'assistant':
                user_prompt = history[-1]['content']
            elif history and history[-1]['role'] == 'user':
                user_prompt = history[-1]['content']
            else:
                # No history, instruct LLM to start the conversation
                user_prompt = ""

            # Generate response
            response = generate_response_with_llm(user_prompt, history, summary, current_character_prompt, MODEL_NAME)

            logger.debug(f"Generated response from LLM for {current_character_name}: {response}")
            if not response:
                response = "I'm sorry, I couldn't process your request."
                logger.warning(f"Empty response from LLM for {current_character_name}. Using default message.")
            else:
                response = remove_timestamps_from_response(response)
                response = remove_leading_name_from_response(response, current_character_name)
                response = clean_response(response)  # Clean unwanted markdown code fences
                logger.debug(f"Processed LLM response for {current_character_name}: {response}")

            # Get current timestamp
            timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

            # Append messages with speaker's name and timestamp
            assistant_message = {
                "role": "assistant",
                "content": response,
                "name": current_character_name,
                "timestamp": timestamp  # Consider storing original message timestamp if needed
            }
            history.append(assistant_message)
            logger.debug(f"Appended assistant message to history: {assistant_message}")

            # Put the new message into the shared_history for automatic updates
            with shared_history_lock:
                shared_history.append(assistant_message)
                logger.debug(f"Shared history updated with new message: {assistant_message}")

            # Add new message to the queue for the front-end to process
            new_message_queue.put(assistant_message)

            # Check if context exceeds maximum length and summarize if necessary
            context_length = len(json.dumps(history))
            if context_length > MAX_CONTEXT_LENGTH:
                summary = summarize_history(history, summary, current_character_prompt, MODEL_NAME, num_recent=DEFAULT_NUMBER_OF_RECENT_MESSAGES_TO_KEEP)
                # Keep only the recent messages
                history = history[-DEFAULT_NUMBER_OF_RECENT_MESSAGES_TO_KEEP:]
                logger.info("Context length exceeded maximum. Summarized the conversation.")

            # Store the updated session data with new summary and history
            store_session_data(session_id_global, history, summary)
            logger.debug("Updated session data stored after auto chat.")

            # Move to next character
            current_index = (current_index + 1) % len(selected_characters_global)
            logger.debug(f"Moving to next character index: {current_index}")

            # Wait for a short delay to simulate conversation flow
            time.sleep(5)  # Adjust the delay as needed
    except Exception as e:
        logger.error(f"Exception in auto_chat thread: {e}")
        logger.debug(traceback.format_exc())
        auto_mode_active = False  # Ensure the flag is reset on exception

# Function to stop auto chat
def stop_auto_chat():
    """Stop the auto chat thread."""
    global auto_mode_active
    if not auto_mode_active:
        logger.info("Auto chat is not active.")
        return (
            gr.update(),  # chatbot remains unchanged
            history_placeholder(),
            summary_placeholder(),
            gr.update(),  # character_info_display remains unchanged
            auto_refresh_trigger  # No change
        )
    auto_mode_active = False
    logger.info("Auto chat stopped.")
    return (
        gr.update(),  # chatbot remains unchanged
        history_placeholder(),
        summary_placeholder(),
        gr.update(),  # character_info_display remains unchanged
        auto_refresh_trigger  # No change
    )

# Function to summarize the conversation history
def summarize_history(history, summary, system_prompt, model, num_recent=DEFAULT_NUMBER_OF_RECENT_MESSAGES_TO_KEEP):
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
    return summary.strip()

# Function to generate a response from the LLM
def generate_response_with_llm(user_prompt: str, history: list, summary: str, character_prompt: str, model: str) -> str:
    """
    Generate a response from the LLM based on the prompt, context (history), and memory (summary).
    This function interacts with the Ollama API, uses proper message formatting, and leverages caching.
    It sends a system message and a single user message containing the conversation history or instructions to start the conversation.
    """
    logger.debug(f"Entered generate_response_with_llm with prompt size {len(user_prompt)}")
    global cache

    character_name = get_character_name(character_prompt)

    cache_key = generate_cache_key(user_prompt, history, summary, character_prompt, model)
    logger.debug(f"Generated cache key: {cache_key}")
    # Attempt to fetch from cache
    with cache_lock:
        if cache and cache_key in cache:
            logger.info(f"Fetching result from cache for prompt: {user_prompt[:50]}...")
            response_content = cache[cache_key]
            logger.debug(f"Cached response: {response_content}")
            return response_content

    # If not cached, call the LLM API
    try:
        # Determine if there is conversation history
        has_history = any(msg['role'] in ['user', 'assistant'] for msg in history)

        if has_history:
            # Format the conversation history
            conversation_history = ""
            for message in history:
                if message['role'] in ['user', 'assistant']:
                    timestamp = message.get('timestamp', '')
                    name = message.get('name', 'Unknown')
                    content = message.get('content', '')
                    conversation_history += f"[{timestamp}] {name}: {content}\n"

            # Construct the user message with history and instruction
            full_user_prompt = f"""
===  
Conversation history:  

{conversation_history}  
===  
As {character_name}, respond to the latest message in a single line using the conversation history for context and guidance.  

Guidelines:  
1. Stay in character as {character_name}.  
2. Use Markdown for formatting. Describe feelings and actions using *cursive* markdown (e.g., *this*).  
3. On your turn, you can choose to either just describe your feelings and actions or also speak.  
4. Keep responses concise yet engaging.
"""
        else:
            # No history, instruct LLM to start the conversation
            full_user_prompt = f"""
As {character_name}, start a conversation with the user.  

Guidelines:  
1. Stay in character as {character_name}.  
2. Use Markdown for formatting. Describe feelings and actions using *cursive* markdown (e.g., *this*).  
3. Keep responses concise yet engaging.
"""

        messages = [
            {"role": "system", "content": character_prompt},
            {"role": "user", "content": full_user_prompt}
        ]

        payload = {
            "model": model,
            "messages": messages
        }

        logger.info(f"Sending request to LLM with model '{model}' and prompt size {len(full_user_prompt)}")
        logger.debug(f"Full payload:\n{json.dumps(payload, indent=2)}")

        headers = {'Content-Type': 'application/json'}
        # Enable streaming for response with a timeout to prevent hanging
        response = requests.post(OLLAMA_URL, data=json.dumps(payload), headers=headers, stream=True, timeout=60)

        logger.debug(f"Response status code: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"Failed to generate response with LLM: HTTP {response.status_code}")
            logger.debug(f"Response content: {response.text}")
            return ""

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

        if not response_content.strip():
            logger.warning("Received empty response from LLM.")
            logger.debug(f"Complete raw response: {raw_response}")
            return ""

        # Cache the result outside the lock to prevent holding the lock during I/O
        if cache:
            with cache_lock:
                logger.debug("Caching the generated response.")
                cache[cache_key] = response_content
            cache.sync()

        logger.debug(f"Generated response from LLM: {response_content.strip()}")
        return response_content.strip()

    except requests.exceptions.Timeout:
        logger.error("LLM API request timed out.")
        return "I'm sorry, the request timed out. Please try again later."
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API request failed: {e}")
        logger.debug(traceback.format_exc())
        return "I'm sorry, there was an error processing your request."
    except Exception as e:
        logger.error(f"Failed to generate response with LLM: {e}")
        logger.debug(traceback.format_exc())
        return ""

# Function to refresh the chat with new messages from the shared_history
def automatic_refresh(history, summary, session_id):
    """Automatically refresh the chatbot with new messages from the shared_history."""
    logger.debug("Refreshing chat with new messages from the shared_history.")
    try:
        new_messages = []
        while not new_message_queue.empty():
            message = new_message_queue.get_nowait()
            new_messages.append(message)

        if not new_messages:
            logger.debug("No new messages to refresh.")
            return (
                gr.update(),  # No update to chatbot
                history,       # history remains the same
                summary        # summary remains the same
            )

        for new_message in new_messages:
            history.append(new_message)
            logger.debug(f"Appended new message from queue: {new_message}")

        formatted_history = format_history_for_display(history)
        # Update session data with new history and summary
        if session_id:
            store_session_data(session_id, history, summary)
            logger.debug("Session data updated with new messages from queue.")

        return (
            gr.update(value=formatted_history),  # Update chatbot
            history,                             # Update history state
            summary                              # Update summary state
        )
    except Exception as e:
        logger.error(f"Error refreshing chat: {e}")
        logger.debug(traceback.format_exc())
        return (
            gr.update(),  # No update to chatbot
            history,       # history remains the same
            summary        # summary remains the same
        )

# Main function to set up Gradio interface
def main():
    global cache, session_id_global, selected_characters_global
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

        with gr.Blocks() as demo:
            # Add custom CSS for slideshow styling
            gr.HTML("""
            <style>
                #slideshow_text {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 80vh;
                    font-size: 2em;
                    text-align: center;
                    padding: 20px;
                }
                #slideshow_container {
                    width: 100%;
                    height: 100%;
                }
            </style>
            """)

            # Add custom JavaScript for keyboard navigation
            gr.HTML("""
            <script>
                // Function to simulate button clicks
                function simulateClick(buttonId) {
                    var button = document.getElementById(buttonId);
                    if(button) {
                        button.click();
                    }
                }

                // Listen for keydown events
                document.addEventListener('keydown', function(event) {
                    // Check if slideshow is visible
                    var slideshow = document.getElementById('slideshow_container');
                    if (slideshow && slideshow.style.display !== 'none') {
                        if(event.key === 'ArrowLeft') {
                            simulateClick('prev_button');
                        } else if(event.key === 'ArrowRight') {
                            simulateClick('next_button');
                        }
                    }
                });
            </script>
            """)

            gr.Markdown("# Multi-Character Chatbot with Automatic Mode")

            # Character info display
            character_info_display = gr.Markdown()

            # Initialize session variables
            history = gr.State([])
            summary = gr.State("")
            assistant_character = gr.State(None)  # State for assistant character

            # Hidden trigger for automatic refresh
            auto_refresh_trigger = gr.State(0)

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

                    # Auto chat controls
                    with gr.Row():
                        selected_characters = gr.CheckboxGroup(
                            choices=character_files,
                            label="Select Characters for Auto Chat"
                        )
                        start_auto_button = gr.Button("Start Auto Chat")
                        stop_auto_button = gr.Button("Stop Auto Chat")

                    # Assistant character selection
                    with gr.Row():
                        assistant_character_dropdown = gr.Dropdown(
                            choices=character_files,
                            label="Select Assistant Character",
                            value=None,
                            interactive=True
                        )

                    # Initialize Chatbot with 'messages' type
                    chatbot = gr.Chatbot(value=[], type='messages')
                    user_input = gr.Textbox(label="Your Message", placeholder="Type your message here and press Enter")

                    # Reset chat button
                    reset_button = gr.Button("Reset Chat")

            # Slideshow Interface Components (Initially Hidden)
            with gr.Row(visible=False, elem_id="slideshow_container") as slideshow_container:
                slideshow_text = gr.Markdown("**Slideshow Display**", visible=True, elem_id="slideshow_text")
                with gr.Row():
                    prev_button = gr.Button("◀️ Previous", elem_id="prev_button")
                    next_button = gr.Button("Next ▶️", elem_id="next_button")
                    back_button = gr.Button("Back to Chat")

            # Slideshow State
            slideshow_index = gr.State(0)

            # Load session data when session ID changes
            def on_session_change(session_id_value):
                logger.debug(f"Session ID changed to: {session_id_value}")
                try:
                    if not session_id_value:
                        logger.info("Session ID is None. Resetting components.")
                        return (
                            gr.update(value=[]),
                            [],
                            "",
                            gr.update(value="**Current Date & Time:** N/A"),
                            "Assistant",   # Update assistant_character state to default
                            gr.update(visible=True),  # Show chat_container
                            gr.update(visible=False),  # Hide slideshow_container
                            0  # Reset auto_refresh_trigger
                        )
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
                        # If no assistant character found in history, default to the first character
                        assistant_char = next(iter(character_prompts))

                    logger.debug(f"Assistant character determined: {assistant_char}")

                    return (
                        gr.update(value=format_history_for_display(history_value)),
                        history_value,
                        summary_value,
                        character_info,
                        assistant_char,          # Update assistant_character state
                        gr.update(visible=True),  # Show chat_container
                        gr.update(visible=False),  # Hide slideshow_container
                        0  # Reset auto_refresh_trigger
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
                        0  # Reset auto_refresh_trigger
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
                    auto_refresh_trigger  # Reset trigger
                ]
            )

            # Handle user input submission
            user_input.submit(
                respond,
                inputs=[user_input, history, summary, session_id_dropdown, assistant_character],
                outputs=[chatbot, history, summary, user_input, session_id_dropdown, assistant_character]
            )

            # Create new session
            new_session_button.click(
                create_new_session,
                inputs=None,
                outputs=[session_id_dropdown, chatbot, history, summary, character_info_display, assistant_character]
            )

            # Delete session
            delete_session_button.click(
                delete_session,
                inputs=session_id_dropdown,
                outputs=[session_id_dropdown, chatbot, history, summary, character_info_display, assistant_character]
            )

            # Reset chat
            reset_button.click(
                reset_chat,
                inputs=session_id_dropdown,
                outputs=[chatbot, history, summary, character_info_display, assistant_character]
            )

            # Start auto chat
            start_auto_button.click(
                start_auto_chat,
                inputs=[selected_characters, session_id_dropdown, auto_refresh_trigger],
                outputs=[chatbot, history, summary, character_info_display, auto_refresh_trigger]
            )

            # Stop auto chat
            stop_auto_button.click(
                stop_auto_chat,
                inputs=None,
                outputs=[chatbot, history, summary, character_info_display, auto_refresh_trigger]
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
                        return gr.update(visible=True), gr.update(visible=False), 0
                    history_value, _ = retrieve_session_data(session_id_value)
                    if not history_value:
                        logger.warning("No history available for slideshow.")
                        return gr.update(visible=True), gr.update(visible=False), 0
                    return (
                        gr.update(visible=False),  # Hide chat_container
                        gr.update(visible=True),   # Show slideshow_container
                        0                               # Reset slideshow index
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
                    if index < 0:
                        index = 0
                    elif index >= len(history):
                        index = len(history) - 1
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

            # Back Button Click
            back_button.click(
                exit_slideshow,
                inputs=None,
                outputs=[chat_container, slideshow_container]
            )

            # Automatic Refresh Function
            def automatic_refresh_wrapper(history_val, summary_val, session_id_val):
                """Wrapper function to call the automatic_refresh and handle outputs."""
                return automatic_refresh(history_val, summary_val, session_id_val)

            # Link the auto_refresh_trigger to the automatic_refresh_renderer
            auto_refresh_trigger.change(
                automatic_refresh_wrapper,
                inputs=[history, summary, session_id_dropdown],
                outputs=[chatbot, history, summary]
            )

            # Start a background thread to monitor the new_message_queue and trigger refresh
            def monitor_new_messages():
                while True:
                    try:
                        # Block until a new message is available
                        new_message = new_message_queue.get()
                        logger.debug("New message received in queue. Triggering auto_refresh.")
                        # Increment the auto_refresh_trigger to trigger the Gradio event
                        # Since we cannot modify gr.State directly, use a separate mechanism
                        # Here, we'll use a Gradio event to handle it
                        # However, Gradio does not support direct triggers from threads
                        # Instead, we can use a polling mechanism or alternative approaches
                        # For simplicity, we'll just log the event
                    except Exception as e:
                        logger.error(f"Error in monitor_new_messages: {e}")
                        logger.debug(traceback.format_exc())

            monitor_thread = threading.Thread(target=monitor_new_messages, daemon=True)
            monitor_thread.start()
            logger.debug("Started monitor_new_messages thread.")

            # Load existing sessions and set default session on app load
            def load_default_session():
                logger.debug("Loading default session on app load.")
                try:
                    existing_sessions = get_existing_sessions()
                    logger.debug(f"Existing sessions on load: {existing_sessions}")
                    if existing_sessions:
                        selected_session_id = existing_sessions[0]
                        history_value, summary_value = retrieve_session_data(selected_session_id)
                    else:
                        # If no existing sessions, create a new one
                        selected_session_id = str(uuid.uuid4())
                        history_value = []
                        summary_value = ""
                        store_session_data(selected_session_id, history_value, summary_value)
                        existing_sessions = [selected_session_id]
                        logger.debug(f"Created new session on load: {selected_session_id}")
                    history_value = add_name_to_history(history_value)
                    formatted_history = format_history_for_display(history_value)
                    # Update character info display
                    character_info = update_character_info()

                    # Determine assistant character from history
                    assistant_char = None
                    for msg in reversed(history_value):
                        if msg['role'] == 'assistant':
                            assistant_char = msg.get('name', None)
                            break

                    if not assistant_char and character_prompts:
                        # If no assistant character found in history, default to the first character
                        assistant_char = next(iter(character_prompts))

                    logger.debug(f"Assistant character on load: {assistant_char}")

                    return (
                        gr.update(choices=existing_sessions, value=selected_session_id),  # session_id_dropdown
                        gr.update(value=format_history_for_display(history_value)),       # chatbot
                        history_value,                                                   # history state
                        summary_value,                                                   # summary state
                        character_info,                                                  # character_info_display
                        assistant_char,                                                  # assistant_character state
                        gr.update(visible=True),                                         # Show chat_container
                        gr.update(visible=False),                                        # Hide slideshow_container
                        0  # Initialize auto_refresh_trigger
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
                        gr.update(visible=True),
                        gr.update(visible=False),
                        0  # Initialize auto_refresh_trigger
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
                    chat_container,
                    slideshow_container,
                    auto_refresh_trigger
                ]
            )

            # Periodically check for new messages and trigger auto_refresh
            def periodic_refresh():
                while True:
                    try:
                        new_message = new_message_queue.get()
                        if new_message:
                            logger.debug("Processing new message for automatic refresh.")
                            # Increment the auto_refresh_trigger safely
                            # Since we cannot modify gr.State directly, use Gradio's API via a queue
                            # Here, we'll simulate an increment by updating the state
                            # Note: This is a workaround; Gradio doesn't support direct state manipulation from threads
                            # One possible way is to use a synchronized variable
                            # However, for simplicity, we'll skip auto-refresh triggering
                            # and assume that the front-end periodically checks for new messages
                            pass
                    except Exception as e:
                        logger.error(f"Error in periodic_refresh: {e}")
                        logger.debug(traceback.format_exc())

            refresh_thread = threading.Thread(target=periodic_refresh, daemon=True)
            refresh_thread.start()
            logger.debug("Started periodic_refresh thread.")

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
