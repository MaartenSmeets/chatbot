# Multi-Character Chatbot with Automatic Mode

This repository contains a Python application that implements a multi-character chatbot with automatic mode using Gradio. The chatbot allows users to interact with multiple AI characters, supports auto-chat mode where characters can converse among themselves, and includes features like session management and a slideshow display of conversation history.

## Features

- **Multi-Character Support**: Interact with multiple AI characters, each with customizable personalities defined in YAML files.
- **Automatic Chat Mode**: Characters can engage in conversations autonomously without user input.
- **Session Management**: Create, delete, and reset chat sessions with persistent conversation history stored in SQLite.
- **Slideshow Mode**: Review conversation history in a slideshow format with navigation controls.
- **User Interface**: Intuitive web interface built with Gradio for seamless interaction.

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **[Ollama](https://ollama.ai/docs/installation)** installed and running at `http://localhost:11434`
- **[Ollama Llama3.1 Model](https://ollama.ai/models/llama2](https://ollama.com/library/llama3.1)**: Ensure the `llama3.1:70b-instruct-q4_K_M` model is available in Ollama.

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/MaartenSmeets/chatbot.git
   cd chatbot
   ```

2. **Set Up a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Python Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Ollama is Running**

   - Start Ollama if it's not already running.
   - Verify it's accessible at `http://localhost:11434`.

5. **Download the Required LLM Model**

   ```bash
   ollama pull llama3.1:70b-instruct-q4_K_M
   ```

6. **Run the Application**

   ```bash
   python multi_agent.py
   ```

   Replace `app.py` with the actual filename of the script.

## Usage

1. **Access the Web Interface**

   - Open your web browser and navigate to `http://localhost:7860`.

2. **Interact with the Chatbot**

   - **Select or Create a Session**: Use the dropdown menu to select an existing session or create a new one.
   - **Enter Your Name**: Provide a name to personalize your interactions.
   - **Choose an Assistant Character**: Select a character to interact with from the dropdown.
   - **Auto Chat Controls**:
     - **Select Characters for Auto Chat**: Choose characters for the auto-chat mode.
     - **Start Auto Chat**: Click "Start Auto Chat" to begin autonomous conversations.
     - **Stop Auto Chat**: Click "Stop Auto Chat" to end the automatic conversation.
   - **Manual Chatting**: Type messages in the input box and press Enter to communicate with the assistant.
   - **Reset Chat**: Use the "Reset Chat" button to clear the current conversation.
   - **Delete Session**: Remove the current session and its history.

3. **Slideshow Mode**

   - **Enter Slideshow**: Click the "Slideshow" button to view conversation history.
   - **Navigate Messages**:
     - **First**: Go to the first message.
     - **Previous**: View the previous message.
     - **Next**: View the next message.
     - **Last**: Go to the last message.
   - **Exit Slideshow**: Click "Back to Chat" to return to the main interface.

## Configuration

- **Character Customization**:
  - Character configurations are stored in the `characters` directory as YAML files.
  - Modify existing files or add new ones to customize character behavior.
- **Logging**:
  - Logs are saved in `app.log`.
  - Adjust logging levels in the script as needed.

## Dependencies

The application relies on the following Python libraries:

- `gradio`
- `requests`

Install all dependencies using:

```bash
pip install -r requirements.txt
```

_**Note**: Ensure that `requirements.txt` includes all the necessary libraries._

## Troubleshooting

- **LLM API Timeout**: If you encounter a timeout error, ensure that Ollama is running and the LLM model is properly installed.
- **Model Not Found**: Verify that the `llama3.1:70b-instruct-q4_K_M` model is correctly installed in Ollama.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).
