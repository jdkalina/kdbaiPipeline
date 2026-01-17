"""
Gradio chat interface for the Q for Mortals RAG system.

Provides a conversational interface for querying Q4M documentation.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import gradio as gr
import httpx
import yaml

from src.api.query import search_similar

logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("config/config.yaml")) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# Load configuration
config = load_config()
chat_config = config.get("chat", {})
llm_config = chat_config.get("llm", {})


def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    """Build a RAG prompt with retrieved context.

    Args:
        query: The user's question.
        context_chunks: List of retrieved document chunks.

    Returns:
        Formatted prompt string for the LLM.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        chapter = chunk.get("chapter", "")
        heading = chunk.get("heading", "")
        text = chunk.get("text", "")

        source = f"{chapter}"
        if heading:
            source += f" > {heading}"

        context_parts.append(f"[Source {i}: {source}]\n{text}")

    context_text = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant that answers questions about the Q programming language and kdb+ database using the Q for Mortals documentation.

Use the following context from the Q for Mortals documentation to answer the user's question. If the context doesn't contain enough information to answer, say so and suggest what topics might be relevant.

Context:
{context_text}

User Question: {query}

Answer:"""

    return prompt


def generate_answer_ollama(prompt: str) -> str:
    """Generate an answer using Ollama.

    Args:
        prompt: The prompt to send to the LLM.

    Returns:
        The generated answer.
    """
    endpoint = llm_config.get("ollama_endpoint", "http://localhost:11434")
    model = llm_config.get("model", "llama3.2")
    temperature = llm_config.get("temperature", 0.7)
    max_tokens = llm_config.get("max_tokens", 1024)

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{endpoint}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "No response generated.")
    except httpx.TimeoutException:
        logger.error("Ollama request timed out")
        return "The request timed out. Please try again or ask a simpler question."
    except httpx.HTTPError as e:
        logger.error(f"Ollama HTTP error: {e}")
        return f"Error connecting to Ollama: {e}"
    except Exception as e:
        logger.error(f"Unexpected error with Ollama: {e}")
        return f"An unexpected error occurred: {e}"


def generate_answer_openai(prompt: str) -> str:
    """Generate an answer using OpenAI API.

    Args:
        prompt: The prompt to send to the LLM.

    Returns:
        The generated answer.
    """
    try:
        import openai
    except ImportError:
        return "OpenAI library not installed. Please run: pip install openai"

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY environment variable not set."

    model = llm_config.get("openai_model", "gpt-4o-mini")
    temperature = llm_config.get("temperature", 0.7)
    max_tokens = llm_config.get("max_tokens", 1024)

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Q/kdb+ programming."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return f"Error with OpenAI: {e}"


def format_citations(context_chunks: list[dict]) -> str:
    """Format source citations as clickable links.

    Args:
        context_chunks: List of retrieved document chunks.

    Returns:
        Formatted markdown string with citations.
    """
    if not context_chunks:
        return ""

    citations = []
    seen_urls = set()  # Avoid duplicate citations

    for i, chunk in enumerate(context_chunks, 1):
        chapter = chunk.get("chapter", "Unknown")
        heading = chunk.get("heading", "")
        url = chunk.get("url", "")

        # Skip duplicates
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        source_name = chapter
        if heading:
            source_name += f" > {heading}"

        if url:
            citations.append(f"[{i}] [{source_name}]({url})")
        else:
            citations.append(f"[{i}] {source_name}")

    if not citations:
        return ""

    return "\n\n---\n**Sources:**\n" + "\n".join(citations)


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Generate an answer using the configured LLM provider.

    Args:
        query: The user's question.
        context_chunks: List of retrieved document chunks.

    Returns:
        The generated answer with source citations.
    """
    if not context_chunks:
        return (
            "I couldn't find any relevant documentation for your question. "
            "Please try rephrasing or ask about a different topic from Q for Mortals."
        )

    prompt = build_rag_prompt(query, context_chunks)
    logger.debug(f"Generated prompt ({len(prompt)} chars)")

    provider = llm_config.get("provider", "ollama")

    if provider == "openai":
        answer = generate_answer_openai(prompt)
    else:
        answer = generate_answer_ollama(prompt)

    # Add source citations to the response
    citations = format_citations(context_chunks)

    return answer + citations


def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve relevant document chunks for a query.

    Args:
        query: The user's question.
        top_k: Number of results to retrieve.

    Returns:
        List of search results with text and metadata.
    """
    try:
        results = search_similar(query_text=query, top_k=top_k)
        logger.info(f"Retrieved {len(results)} chunks for query")
        return results
    except Exception as e:
        logger.error(f"Failed to retrieve context: {e}")
        return []


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks for display.

    Args:
        results: List of search results.

    Returns:
        Formatted string showing retrieved context.
    """
    if not results:
        return "*No relevant documentation found.*"

    formatted_parts = []
    for i, result in enumerate(results, 1):
        chapter = result.get("chapter", "Unknown")
        heading = result.get("heading", "")
        text = result.get("text", "")[:500]  # Truncate long chunks
        url = result.get("url", "")

        part = f"**[{i}] {chapter}**"
        if heading:
            part += f" - {heading}"
        part += f"\n{text}"
        if len(result.get("text", "")) > 500:
            part += "..."
        if url:
            part += f"\n[Source]({url})"

        formatted_parts.append(part)

    return "\n\n---\n\n".join(formatted_parts)


def respond(message: str, history: list[tuple[str, str]]) -> tuple[str, str]:
    """Process a user message and generate a response.

    Args:
        message: The user's message.
        history: List of (user, assistant) message tuples.

    Returns:
        Tuple of (assistant response, formatted context).
    """
    if not message or not message.strip():
        return "Please enter a question about Q for Mortals.", ""

    logger.info(f"Received message: {message[:50]}...")

    # Retrieve relevant context from KDB.AI
    results = retrieve_context(message)
    context_display = format_context(results)

    # Generate answer using LLM
    response = generate_answer(message, results)

    return response, context_display


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

        with gr.Row():
            with gr.Column(scale=2):
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

            with gr.Column(scale=1):
                gr.Markdown("### Retrieved Context")
                context_display = gr.Markdown(
                    value="*Context from Q for Mortals documentation will appear here after you ask a question.*",
                    label="Context",
                )

        # Event handlers
        def user_submit(message: str, history: list) -> tuple:
            """Handle user message submission."""
            if not message.strip():
                return "", history, ""
            response, context = respond(message, history)
            history = history + [(message, response)]
            return "", history, context

        msg.submit(
            user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, context_display],
        )

        submit_btn.click(
            user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, context_display],
        )

        clear_btn.click(
            lambda: ([], "", "*Context will appear here after you ask a question.*"),
            outputs=[chatbot, msg, context_display],
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
