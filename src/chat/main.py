"""
Gradio chat interface for the Q for Mortals RAG system.

Provides a conversational interface for querying Q4M documentation.
"""

import logging
from pathlib import Path
from typing import Optional

import gradio as gr
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("config/config.yaml")) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# Load configuration
config = load_config()
chat_config = config.get("chat", {})


def respond(message: str, history: list[tuple[str, str]]) -> str:
    """Process a user message and generate a response.

    Args:
        message: The user's message.
        history: List of (user, assistant) message tuples.

    Returns:
        The assistant's response.
    """
    if not message or not message.strip():
        return "Please enter a question about Q for Mortals."

    # Placeholder response - will be replaced with actual RAG in chat-02/chat-03
    logger.info(f"Received message: {message[:50]}...")
    return f"You asked: '{message}'\n\nThis is a placeholder response. The RAG integration will be added in the next stories."


def create_interface() -> gr.Blocks:
    """Create the Gradio chat interface.

    Returns:
        Gradio Blocks application.
    """
    with gr.Blocks(
        title="Q for Mortals Assistant",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # Q for Mortals Assistant

            Ask questions about Q/kdb+ programming using the Q for Mortals documentation.
            """
        )

        chatbot = gr.Chatbot(
            label="Chat",
            height=400,
            show_copy_button=True,
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your question",
                placeholder="Ask a question about Q/kdb+...",
                scale=4,
                show_label=False,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("Clear Chat")

        # Event handlers
        def user_submit(message: str, history: list) -> tuple:
            """Handle user message submission."""
            if not message.strip():
                return "", history
            response = respond(message, history)
            history = history + [(message, response)]
            return "", history

        msg.submit(
            user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )

        submit_btn.click(
            user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )

        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, msg],
        )

        gr.Markdown(
            """
            ---
            *Powered by KDB.AI semantic search and Q for Mortals documentation*
            """
        )

    return demo


def main():
    """Run the Gradio chat interface."""
    # Configure logging
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_format = log_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.basicConfig(level=log_level, format=log_format)

    host = chat_config.get("host", "0.0.0.0")
    port = chat_config.get("port", 7860)

    logger.info(f"Starting Q4M Chat Interface on {host}:{port}")

    demo = create_interface()
    demo.launch(
        server_name=host,
        server_port=port,
        share=False,
    )


if __name__ == "__main__":
    main()
