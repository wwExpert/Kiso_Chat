# Kiso_Chat

A Streamlit chat application powered by OpenAI's APIs and LangChain.

## Features

- Secure handling of the API key via environment variables.
- Asynchronous processing support.
- Configurable model parameters (temperature, max tokens, top p).
- Multiple chat sessions using a chat ID.
- Streaming of assistant responses.
- Agentic mode for autonomous background tasks.
- Summarize chats into bullet points with a single click.
- Optional message optimization (MCO) to refine user prompts.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/wwExpert/Kiso_Chat.git](https://github.com/wwExpert/Kiso_Chat.git)
    cd Kiso_Chat
    ```

2.  **Install dependencies:**
    Make sure you have Python 3.8+ installed. Then, install the required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set your API Key:**
    The application requires an OpenAI API key. You can enter it directly in the sidebar of the running application.

## How to Run

Run the app with:

```bash
streamlit run kiso_chatgpt.py
```
