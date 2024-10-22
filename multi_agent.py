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

# Initialize a lock for session data to prevent race conditions
session_lock = threading.Lock()

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

# Function to ensure the 'characters' directory exists and add default characters
def ensure_character_directory():
    """Ensure the 'characters' directory exists, create it if not, and add default characters."""
    if not os.path.exists(CHARACTER_DIR):
        os.makedirs(CHARACTER_DIR)
        logger.info(f"Created '{CHARACTER_DIR}' directory.")
        # Add default character files if necessary
        default_characters = {
            'Lily.txt': "You are Lily, a friendly and energetic individual who loves to engage in deep conversations about personal growth and happiness. Your presence is uplifting, and you always aim to bring positivity to every interaction.",
            'Niko.txt': "You are Niko, a thoughtful and introspective person with a passion for technology and innovation. You enjoy discussing complex topics and providing insightful perspectives on various subjects.",
            'Serena.txt': "You are Serena, a calm and composed mentor with expertise in emotional intelligence and mindfulness. You guide others with patience and wisdom, helping them navigate through their challenges with grace."
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

# Function to load all character prompts
def load_all_character_prompts():
    """Load all character prompts and combine them into a single system prompt."""
    character_files = get_character_files()
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
    
    # Add debug log to confirm data is being stored
    logger.debug(f"Stored session data for session_id: {session_id} with history: {history} and summary: {summary}")

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
    # Return exactly 6 outputs corresponding to Gradio components
    return (
        gr.update(choices=existing_sessions, value=new_session_id),  # session_id_dropdown
        gr.update(value=format_history_for_display(history)),       # chatbot
        history,                                                   # history state
        summary,                                                   # summary state
        gr.update(value=character_info),                           # character_info_display
        assistant_char                                             # assistant_character state
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
        logger.debug(f"Created new session: {new_session[0].kwargs['value']}")
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
        history,                                             # history
        summary,                                             # summary
        "",                                                  # user_input
        gr.update(value=character_info),                     # character_info_display
        assistant_char                                       # assistant_character state
    )

# Queue for new messages to be processed by Gradio
new_message_queue = queue.Queue()

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
            assistant_character                                      # assistant_character state
        )

    if not session_id:
        logger.warning("No session ID selected.")
        return (
            gr.update(value=format_history_for_display(history)),  # chatbot
            history,                                                 # history
            summary,                                                 # summary
            "",                                                      # user_input
            assistant_character                                      # assistant_character state
        )

    # Get current timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

    with session_lock:
        if auto_mode_active:
            # In Auto Chat Mode: Directly add user message to history
            user_message = {
                "role": "user",
                "content": user_input,
                "name": user_name,
                "timestamp": timestamp
            }
            history.append(user_message)
            logger.debug(f"Appended user message to history in auto mode: {user_message}")

            # Store the updated session data
            store_session_data(session_id, history, summary)
            logger.debug("Updated session data stored in auto mode.")

            # Return updated components
            return (
                gr.update(value=format_history_for_display(history)),  # chatbot
                history,                                             # history
                summary,                                             # summary
                "",                                                  # user_input
                assistant_character                                  # assistant_character state
            )
        else:
            # Normal Mode: Existing behavior of sending message to LLM
            # Append user message to history with custom name and current timestamp
            user_message = {
                "role": "user",
                "content": user_input,
                "name": user_name,
                "timestamp": timestamp
            }
            history.append(user_message)
            logger.debug(f"Appended user message to history: {user_message}")

            # Store the updated session data
            store_session_data(session_id, history, summary)
            logger.debug("Updated session data stored.")

    # Generate response from LLM outside the lock to prevent blocking
    character_name = assistant_character if assistant_character else "Assistant"
    character_prompt = character_prompts.get(character_name, "") if character_prompts else ""

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

        # Store the updated session data with the character's response
        store_session_data(session_id, history, summary)
        logger.debug("Character response stored in session data.")

    # Return updated components
    return (
        gr.update(value=format_history_for_display(history)),  # chatbot
        history,                                             # history
        summary,                                             # summary
        "",                                                  # user_input
        assistant_character                                  # assistant_character state
    )

# Function to start auto chat
def start_auto_chat(selected_characters, session_id, character_prompts):
    """Start the auto chat thread."""
    global auto_mode_active, selected_characters_global, session_id_global, auto_chat_thread

    logger.debug(f"Start Auto Chat called with characters: {selected_characters} and session_id: {session_id}")

    if not selected_characters:
        logger.warning("No characters selected for auto chat.")
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), "Assistant"
        )

    if not session_id:
        logger.info("No session selected for auto chat. Creating a new session.")
        # Create a new session
        create_new_session(character_prompts)
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
            assistant_char
        )

    auto_mode_active = True
    selected_characters_global = selected_characters
    session_id_global = session_id

    # Start the auto chat thread
    auto_chat_thread = threading.Thread(target=auto_chat, args=(selected_characters, session_id_global, character_prompts), daemon=True)
    auto_chat_thread.start()
    logger.info("Auto chat thread started.")

    return (
        gr.update(value=format_history_for_display(history)),
        history,
        summary,
        gr.update(value=character_info),
        assistant_char
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
            "Assistant"
        )
    auto_mode_active = False
    logger.info("Auto chat stopped.")
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        "Assistant"
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

    # Generate a unique cache key based on the current state
    cache_key = generate_cache_key(user_prompt, history, summary, character_prompt, model)
    logger.debug(f"Generated cache key: {cache_key}")
    
    # Attempt to fetch from cache
    with cache_lock:
        if cache and cache_key in cache:
            logger.info(f"Fetching result from cache for prompt: {user_prompt[:50]}...")
            response_content = cache[cache_key]
            logger.debug(f"Cached response: {response_content}")
            return response_content

    try:
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
                    conversation_history += f"[{timestamp}] {name}: {content}\n"

            # User prompt with context and guidelines
            full_user_prompt = f"""
===  
Conversation history:  

{conversation_history}  
===  
As {character_name}, respond to the latest message in a single line using the conversation history for context and guidance.  

Guidelines:  
1. Stay in character as {character_name}.  
2. Use Markdown for formatting but do not mention markdown. Describe feelings and actions using *cursive* markdown (e.g., *this*).  
3. On your turn, you can choose to either just describe your feelings and actions or also speak.  
4. Keep responses concise yet engaging.
5. Do not start with Name: because the message will already include your name.
"""
        else:
            # If there's no history, send a generic start-conversation prompt
            logger.debug("No conversation history found, sending initial conversation start prompt.")
            full_user_prompt = f"""
As {character_name}, start a conversation with the user.  

Guidelines:  
1. Stay in character as {character_name}.  
2. Use Markdown for formatting but do not mention markdown. Describe feelings and actions using *cursive* markdown (e.g., *this*).  
3. Keep responses concise yet engaging.
4. Do not start with Name: because the message will already include your name.
"""

        # Prepare the system and user messages for the LLM
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
        # Enable streaming response with a timeout
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

        # Check if the response content is empty
        if not response_content.strip():
            logger.warning("Received empty response from LLM.")
            logger.debug(f"Complete raw response: {raw_response}")
            return ""

        # Cache the result if caching is enabled
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

# Function to handle automatic chat in background
def auto_chat(selected_characters, session_id, character_prompts):
    """Background thread function to handle automatic chatting."""
    global auto_mode_active
    current_index = 0
    try:
        logger.debug(f"Auto chat started with session_id: {session_id}")

        while auto_mode_active:
            if not selected_characters:
                logger.warning("No characters selected for auto chat.")
                break

            current_character_name = selected_characters[current_index]
            current_character_prompt = character_prompts.get(current_character_name, "")
            logger.debug(f"Current character for auto chat: {current_character_name}")

            # Acquire the lock to safely access and modify the history
            with session_lock:
                history, summary = retrieve_session_data(session_id)
                history = add_name_to_history(history)

                # Prepare user_prompt from the last message or default
                if history and history[-1]['role'] == 'assistant':
                    user_prompt = history[-1]['content']
                elif history and history[-1]['role'] == 'user':
                    user_prompt = history[-1]['content']
                else:
                    # No history, instruct LLM to start the conversation
                    user_prompt = ""

            # Generate response from LLM
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

            # Prepare assistant message
            assistant_message = {
                "role": "assistant",
                "content": response,
                "name": current_character_name,
                "timestamp": timestamp
            }

            # Acquire the lock to safely append and store the assistant message
            with session_lock:
                history.append(assistant_message)
                new_message_queue.put(assistant_message)
                logger.debug(f"Appended assistant message to history: {assistant_message}")

                # Store the updated session data with the character's response
                store_session_data(session_id, history, summary)
                logger.debug("Character response stored in session data.")

                # Check if context exceeds maximum length and summarize if necessary
                context_length = len(json.dumps(history))
                if context_length > MAX_CONTEXT_LENGTH:
                    summary = summarize_history(history, summary, current_character_prompt, MODEL_NAME, num_recent=DEFAULT_NUMBER_OF_RECENT_MESSAGES_TO_KEEP)
                    # Keep only the recent messages
                    history = history[-DEFAULT_NUMBER_OF_RECENT_MESSAGES_TO_KEEP:]
                    logger.info("Context length exceeded maximum. Summarized the conversation.")

                    # Store the updated session data with new summary and history
                    store_session_data(session_id, history, summary)
                    logger.debug("Updated session data stored after auto chat.")

            # Move to next character
            current_index = (current_index + 1) % len(selected_characters)
            logger.debug(f"Moving to next character index: {current_index}")

            # Sleep to simulate conversation flow
            time.sleep(5)  # Adjust the delay as needed
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

                    # Auto chat controls
                    with gr.Row():
                        selected_characters = gr.CheckboxGroup(
                            choices=character_names,  # Updated to show only names
                            label="Select Characters for Auto Chat"
                        )
                        start_auto_button = gr.Button("Start Auto Chat")
                        stop_auto_button = gr.Button("Stop Auto Chat")

                    # Assistant character selection
                    with gr.Row():
                        assistant_character_dropdown = gr.Dropdown(
                            choices=character_names,  # Updated to show only names
                            label="Select Assistant Character",
                            value=None,
                            interactive=True
                        )

                    # User name input
                    with gr.Row():
                        user_name_input = gr.Textbox(
                            label="Your Name",
                            value="User",
                            placeholder="Enter your name",
                            interactive=True
                        )
                        # Update user_name state when user_name_input changes
                        user_name_input.change(
                            lambda x: x.strip() if x.strip() else "User",
                            inputs=user_name_input,
                            outputs=user_name
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
                        create_new_session(character_prompts)
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
                    
                    return (
                        gr.update(choices=existing_sessions, value=selected_session_id),  # session_id_dropdown
                        gr.update(value=formatted_history),  # chatbot
                        history_value,  # history state
                        summary_value,  # summary state
                        gr.update(value=character_info),  # character_info_display
                        assistant_char,  # assistant_character state
                        user_name,        # user_name state
                        gr.update(visible=True),  # Show chat_container
                        gr.update(visible=False),  # Hide slideshow_container
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
                            gr.update(value=[]),
                            [],
                            "",
                            gr.update(value="**Current Date & Time:** N/A"),
                            "Assistant",   # Update assistant_character state to default
                            gr.update(visible=True),  # Show chat_container
                            gr.update(visible=False),  # Hide slideshow_container
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
                outputs=[chatbot, history, summary, user_input, assistant_character]
            )

            # Event: Create new session
            def handle_create_new_session():
                return create_new_session(character_prompts)

            new_session_button.click(
                handle_create_new_session,
                inputs=None,
                outputs=[session_id_dropdown, chatbot, history, summary, character_info_display, assistant_character]
            )

            # Event: Delete session
            def handle_delete_session(session_id_value):
                return delete_session(session_id_value, character_prompts)

            delete_session_button.click(
                handle_delete_session,
                inputs=session_id_dropdown,
                outputs=[session_id_dropdown, chatbot, history, summary, character_info_display, assistant_character]
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
            def handle_start_auto_chat(selected_characters_value, session_id_value, history_value):
                """Generator function for auto chat, streaming new messages to the front-end."""

                logger.debug(f"Starting auto-chat for session_id: {session_id_value} with selected_characters: {selected_characters_value}")

                # Start the auto chat thread
                start_auto_chat(selected_characters_value, session_id_value, character_prompts)

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

                # Initial yield to set up the session in the front-end
                yield (
                    gr.update(value=formatted_history),  # chatbot
                    history_fetched,                     # history state
                    summary_fetched,                     # summary state
                    gr.update(value=character_info),     # character_info_display
                    assistant_char                      # assistant_character state
                )

                # Continuously yield updates as new messages arrive
                while auto_mode_active:
                    try:
                        new_message = new_message_queue.get(timeout=1)  # Blocking wait for new message

                        logger.debug(f"New auto-chat message received: {new_message}")

                        with session_lock:
                            # Fetch the latest history from the database
                            history_fetched, summary_fetched = retrieve_session_data(session_id_value)
                            history_fetched = add_name_to_history(history_fetched)
                            formatted_history = format_history_for_display(history_fetched)
                            character_info = update_character_info()

                            # Determine assistant character from updated history
                            assistant_char = None
                            for msg in reversed(history_fetched):
                                if msg['role'] == 'assistant':
                                    assistant_char = msg.get('name', None)
                                    break

                            if not assistant_char and character_prompts:
                                assistant_char = next(iter(character_prompts))

                        # Yield the updated history to the front-end
                        yield (
                            gr.update(value=formatted_history),  # chatbot
                            history_fetched,                     # history state
                            summary_fetched,                     # summary state
                            gr.update(value=character_info),     # character_info_display
                            assistant_char                      # assistant_character state
                        )
                    except queue.Empty:
                        continue  # No new messages, continue waiting

            start_auto_button.click(
                handle_start_auto_chat,
                inputs=[selected_characters, session_id_dropdown, history],  # Pass history as input
                outputs=[chatbot, history, summary, character_info_display, assistant_character],
                queue=True  # Enable Gradio's task queue for async streaming
            )

            # Event: Stop auto chat
            def handle_stop_auto_chat():
                return stop_auto_chat()

            stop_auto_button.click(
                handle_stop_auto_chat,
                inputs=None,
                outputs=[chatbot, history, summary, character_info_display, assistant_character]
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
