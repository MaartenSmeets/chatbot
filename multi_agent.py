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
MODEL_NAME = "llama3.1:70b-instruct-q4_K_M"      # The model version
CHARACTER_DIR = 'characters'                     # Directory where character text files are stored
LOG_FILE = 'app.log'                             # Log file path

# Summarization settings (configurable)
MAX_CONTEXT_LENGTH = 400000  # Max context length before summarizing
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
def generate_cache_key(user_prompt: str, history: list, summary: str, system_prompt: str, model: str, is_decision: bool = False) -> str:
    """
    Generate a unique cache key based on the user prompt, history, summary, system prompt, model, and decision flag.
    """
    history_str = json.dumps(history)
    history_hash = hashlib.sha256(history_str.encode('utf-8')).hexdigest()
    key_components = f"{model}:{system_prompt}:{summary}:{history_hash}:{user_prompt}:{is_decision}"
    cache_key = hashlib.sha256(key_components.encode('utf-8')).hexdigest()
    return cache_key

# Function to remove timestamps from the assistant's response
def remove_timestamps_from_response(response):
    """Remove timestamps at the beginning of the response."""
    # Remove timestamps like '[20-10-2024 13:45] ' at the beginning of the response
    return re.sub(r'^\[\d{2}-\d{2}-\d{4} \d{2}:\d{2}\]\s*', '', response)

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

# Function to extract character name from file
def get_character_name(character_file_or_prompt):
    """Extract character name from character file name or prompt."""
    if isinstance(character_file_or_prompt, str):
        if character_file_or_prompt.endswith('.txt'):
            return os.path.splitext(character_file_or_prompt)[0]
        else:
            # If it's a prompt, try to extract the name from "You are [Name]" pattern
            match = re.search(r"You are (\w+)", character_file_or_prompt)
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
    # Return the updated session dropdown and select the new session
    return (
        gr.update(choices=existing_sessions, value=new_session_id),  # session_id_dropdown
        gr.update(value=[]),                                       # chatbot
        history,                                                   # history
        summary,                                                   # summary
        character_info,                                           # character_info_display
        new_session_id                                            # session_id state
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

    # Return the updated session dropdown and related components, including session_id state
    return (
        gr.update(choices=existing_sessions, value=selected_session_id),  # session_id_dropdown
        gr.update(value=format_history_for_display(history)),            # chatbot
        history,                                                        # history
        summary,                                                        # summary
        character_info,                                                # character_info_display
        selected_session_id                                            # session_id state
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
    return (
        gr.update(value=[]),                # chatbot
        history,                            # history
        summary,                            # summary
        character_info,                     # character_info_display
        session_id                         # session_id state
    )

# Function to handle user input
def respond(user_input, history, summary, session_id):
    """Handle user input and append it to the chat history."""
    logger.debug(f"User input: {user_input}")
    # Check for empty user input
    if not user_input.strip():
        logger.warning("Received empty user input.")
        return (
            gr.update(value=format_history_for_display(history)),  # chatbot
            history,                                               # history
            summary,                                               # summary
            gr.update(value=''),                                  # user_input
            session_id                                             # session_id state
        )

    if not session_id:
        logger.warning("No session ID selected.")
        return (
            gr.update(value=format_history_for_display(history)),  # chatbot
            history,                                               # history
            summary,                                               # summary
            gr.update(value=''),                                  # user_input
            session_id                                             # session_id state
        )

    # Append user message to history with 'User' as the name and current timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")
    history.append({
        "role": "user",
        "content": user_input,
        "name": "User",
        "timestamp": timestamp
    })

    # Store the updated session data
    store_session_data(session_id, history, summary)
    logger.debug("Updated session data stored.")

    # Generate response from LLM
    character_prompts = load_all_character_prompts()
    response = generate_response_with_llm(user_input, history, summary, character_prompts, MODEL_NAME)

    if not response:
        response = "I'm sorry, I couldn't process your request."
    else:
        response = remove_timestamps_from_response(response)

    # Append assistant message to history
    assistant_message = {
        "role": "assistant",
        "content": response,
        "name": "Assistant",
        "timestamp": timestamp
    }
    history.append(assistant_message)

    # Store the updated session data with the assistant's response
    store_session_data(session_id, history, summary)
    logger.debug("Assistant response stored in session data.")

    return (
        gr.update(value=format_history_for_display(history)),  # chatbot
        history,                                               # history
        summary,                                               # summary
        gr.update(value=''),                                  # user_input
        session_id                                             # session_id state
    )
    
# Message queue for auto chat
message_queue = queue.Queue()

# Global state to control auto mode
auto_mode_active = False
selected_characters_global = []
session_id_global = None
auto_chat_thread = None

# Function to start auto chat
def start_auto_chat(selected_characters, session_id):
    """Start the auto chat thread."""
    global auto_mode_active, selected_characters_global, session_id_global, auto_chat_thread, cache, character_prompts

    if not selected_characters:
        logger.warning("No characters selected for auto chat.")
        return []  # Return empty list instead of None

    if not session_id:
        logger.info("No session selected for auto chat. Creating a new session.")
        # Create a new session and update the session_id
        updated_components = create_new_session()
        # Unpack the returned components
        session_id_dropdown_update, chatbot_update, history, summary, character_info, new_session_id = updated_components
        logger.debug(f"Created and selected new session ID: {new_session_id}")
        # Update the session_id_global with the new session_id
        session_id = new_session_id

    if auto_mode_active:
        logger.info("Auto chat is already active.")
        return []  # Return empty list instead of None

    auto_mode_active = True
    selected_characters_global = selected_characters
    session_id_global = session_id

    # Make sure character prompts are loaded
    character_prompts = load_all_character_prompts()

    # Start the auto chat thread
    auto_chat_thread = threading.Thread(target=auto_chat, daemon=True)
    auto_chat_thread.start()
    logger.info("Auto chat thread started.")

    return []

def auto_chat():
    global auto_mode_active, selected_characters_global, session_id_global, cache, character_prompts
    current_index = 0
    try:
        # Retrieve history and summary at the start
        history, summary = retrieve_session_data(session_id_global)

        while auto_mode_active:
            if not selected_characters_global:
                logger.warning("No characters selected for auto chat.")
                break

            current_character_file = selected_characters_global[current_index]
            current_character_name = get_character_name(current_character_file)
            current_character_prompt = character_prompts[current_character_name]

            # Prepare user_prompt from the last message or default
            if history and history[-1]['role'] == 'assistant':
                user_prompt = history[-1]['content']
            elif history and history[-1]['role'] == 'user':
                user_prompt = history[-1]['content']
            else:
                user_prompt = "Hello"

            # Make decision to participate
            decision = generate_response_with_llm(user_prompt, history, summary, current_character_prompt, MODEL_NAME, is_decision=True)
            logger.debug(f"Decision response from LLM for {current_character_name}: {decision}")
            decision = decision.lower().strip()

            if 'yes' in decision:
                # Generate response
                response = generate_response_with_llm(user_prompt, history, summary, current_character_prompt, MODEL_NAME)

                logger.debug(f"Generated response from LLM for {current_character_name}: {response}")
                if not response:
                    response = "I'm sorry, I couldn't process your request."
                else:
                    response = remove_timestamps_from_response(response)

                # Get current timestamp
                timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

                # Append messages with speaker's name and timestamp
                assistant_message = {
                    "role": "assistant",
                    "content": f"{response}",
                    "name": current_character_name,
                    "timestamp": timestamp
                }
                history.append(assistant_message)

                # Put the new message into the queue
                message_queue.put(assistant_message)

                logger.debug(f"{current_character_name} contributed to the conversation.")
            else:
                logger.debug(f"{current_character_name} decided to pass their turn.")

            # Check if context exceeds maximum length and summarize if necessary
            context_length = len(json.dumps(history))
            if context_length > MAX_CONTEXT_LENGTH:
                summary = summarize_history(history, summary, current_character_prompt, MODEL_NAME, num_recent=DEFAULT_NUMBER_OF_RECENT_MESSAGES_TO_KEEP)
                # Keep only the recent messages
                history = history[-DEFAULT_NUMBER_OF_RECENT_MESSAGES_TO_KEEP:]
                logger.info("Context length exceeded maximum. Summarized the conversation.")

            # Store the updated session data with new summary and history
            store_session_data(session_id_global, history, summary)

            # Move to next character
            current_index = (current_index + 1) % len(selected_characters_global)

            # Wait for a short delay to simulate conversation flow
            time.sleep(5)  # Adjust the delay as needed
    except Exception as e:
        logger.error(f"Exception in auto_chat thread: {e}")
        logger.debug(traceback.format_exc())
        auto_mode_active = False  # Ensure the flag is reset on exception
    # Start the auto chat thread
    auto_chat_thread = threading.Thread(target=auto_chat, daemon=True)
    auto_chat_thread.start()
    logger.info("Auto chat thread started.")

    return []  # No outputs

# Function to stop auto chat
def stop_auto_chat():
    """Stop the auto chat thread."""
    global auto_mode_active
    if not auto_mode_active:
        logger.info("Auto chat is not active.")
        return []  # No outputs
    auto_mode_active = False
    logger.info("Auto chat stopped.")
    return []  # Return empty list to match outputs=[]

# Function to refresh the chat with new messages from the queue
def refresh_chat(history, chatbot, summary):
    """Refresh the chatbot with new messages from the queue."""
    try:
        updated = False
        while not message_queue.empty():
            new_message = message_queue.get_nowait()
            history.append(new_message)
            updated = True
            logger.debug(f"New message from auto chat: {new_message}")
        if updated:
            formatted_history = format_history_for_display(history)
            # Update session data with new history and summary
            if session_id_global:
                store_session_data(session_id_global, history, summary)
            return gr.update(value=formatted_history), history, summary
        else:
            return gr.update(), history, summary
    except Exception as e:
        logger.error(f"Error refreshing chat: {e}")
        logger.debug(traceback.format_exc())
        return gr.update(), history, summary

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
def generate_response_with_llm(user_prompt: str, history: list, summary: str, character_prompt: str, model: str, is_decision: bool = False) -> str:
    """
    Generate a response from the LLM based on the prompt, context (history), and memory (summary).
    This function interacts with the Ollama API, uses proper message formatting, and leverages caching.
    It sends a system message and a single user message containing the conversation history.
    """
    logger.debug(f"Entered generate_response_with_llm with prompt size {len(user_prompt)}")
    global cache

    character_name = get_character_name(character_prompt)

    cache_key = generate_cache_key(user_prompt, history, summary, character_prompt, model, is_decision)
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
        # Format the conversation history
        conversation_history = ""
        for message in history:
            if message['role'] in ['user', 'assistant']:
                timestamp = message.get('timestamp', '')
                name = message.get('name', 'Unknown')
                content = message.get('content', '')
                conversation_history += f"[{timestamp}] {name}: {content}\n"

        # Construct the user message with history and instruction
        if is_decision:
            full_user_prompt = f"""
The conversation so far is:

{conversation_history}

As {character_name}, based on your personality and the conversation context, do you want to contribute to the conversation? Respond only with 'yes' if you want to say something, or only with 'no' if you want to pass. The default is 'yes'.
"""
        else:
            full_user_prompt = f"""
Conversation history:

{conversation_history}

As {character_name}, respond to the latest message:

{user_prompt}

Guidelines:
1. Stay in character as {character_name}.
2. Use Markdown for formatting (e.g., **bold**, *italic*, `code`).
3. Describe appearances, feelings, or thoughts using square brackets [like this].
4. Keep responses concise yet engaging.
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
        
# Main function to set up Gradio interface
def main():
    global cache, session_id_global
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
            gr.Markdown("# Multi-Character Chatbot with Automatic Mode")

            # Character info display
            character_info_display = gr.Markdown()

            # Initialize session variables
            session_id = gr.State(None)
            history = gr.State([])
            summary = gr.State("")

            with gr.Row():
                session_id_dropdown = gr.Dropdown(
                    choices=get_existing_sessions(),
                    label="Select Session",
                    value=None,
                    interactive=True
                )
                new_session_button = gr.Button("Create New Session")
                delete_session_button = gr.Button("Delete Session")

            # Auto chat controls
            with gr.Row():
                selected_characters = gr.CheckboxGroup(
                    choices=character_files,
                    label="Select Characters for Auto Chat"
                )
                start_auto_button = gr.Button("Start Auto Chat")
                stop_auto_button = gr.Button("Stop Auto Chat")

            # Initialize Chatbot with 'messages' type
            chatbot = gr.Chatbot(value=[], type='messages')
            user_input = gr.Textbox(label="Your Message", placeholder="Type your message here and press Enter")

            # Reset chat button
            reset_button = gr.Button("Reset Chat")

            # Refresh chat button (manual)
            refresh_button = gr.Button("Refresh Chat")

            # Load session data when session ID changes
            def on_session_change(session_id_value):
                logger.debug(f"Session ID changed to: {session_id_value}")
                if not session_id_value:
                    return (
                        gr.update(value=[]),
                        [],
                        "",
                        "**Current Date & Time:** N/A",
                        None  # Update session_id state to None
                    )
                history_value, summary_value = retrieve_session_data(session_id_value)
                history_value = add_name_to_history(history_value)
                formatted_history = format_history_for_display(history_value)
                character_info = update_character_info()
                return (
                    gr.update(value=formatted_history),
                    history_value,
                    summary_value,
                    character_info,
                    session_id_value  # Update session_id state
                )

            session_id_dropdown.change(
                on_session_change,
                inputs=session_id_dropdown,
                outputs=[chatbot, history, summary, character_info_display, session_id]
            )

            # Handle user input submission
            user_input.submit(
                respond,
                inputs=[user_input, history, summary, session_id],
                outputs=[chatbot, history, summary, user_input, session_id]
            )

            # Create new session
            new_session_button.click(
                create_new_session,
                inputs=None,
                outputs=[session_id_dropdown, chatbot, history, summary, character_info_display, session_id]
            )

            # Delete session
            delete_session_button.click(
                delete_session,
                inputs=session_id_dropdown,
                outputs=[session_id_dropdown, chatbot, history, summary, character_info_display, session_id]
            )

            # Reset chat
            reset_button.click(
                reset_chat,
                inputs=session_id_dropdown,
                outputs=[chatbot, history, summary, character_info_display, session_id]
            )

            # Start auto chat
            start_auto_button.click(
                start_auto_chat,
                inputs=[selected_characters, session_id],
                outputs=[]  # Specify empty list to indicate no outputs
            )

            # Stop auto chat
            stop_auto_button.click(
                stop_auto_chat,
                inputs=None,
                outputs=[]  # Specify empty list to indicate no outputs
            )

            # Refresh chat button (manual)
            refresh_button.click(
                refresh_chat,
                inputs=[history, chatbot, summary],
                outputs=[chatbot, history, summary]
            )

            # On app load, load existing sessions and set default session if available
            def load_default_session():
                existing_sessions = get_existing_sessions()
                logger.debug(f"Loading default session")
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
                    logger.debug(f"Created new session: {selected_session_id}")
                history_value = add_name_to_history(history_value)
                formatted_history = format_history_for_display(history_value)
                # Update character info display
                character_info = update_character_info()
                # Return components, including session_id state
                return (
                    gr.update(choices=existing_sessions, value=selected_session_id),  # session_id_dropdown
                    gr.update(value=formatted_history),                              # chatbot
                    history_value,                                                  # history state
                    summary_value,                                                  # summary state
                    character_info,                                                 # character_info_display
                    selected_session_id                                             # session_id state
                )

            demo.load(
                load_default_session,
                inputs=None,
                outputs=[session_id_dropdown, chatbot, history, summary, character_info_display, session_id]
            )

        logger.info("Launching Gradio app.")
        # Launch Gradio with share=False for local access
        try:
            demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
        except Exception as e:
            logger.error(f"Failed to launch Gradio app: {e}")
            logger.debug(traceback.format_exc())

    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.debug(traceback.format_exc())
