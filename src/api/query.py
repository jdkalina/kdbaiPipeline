"""
Query processing for the Q for Mortals RAG API.

This module provides functions for embedding user queries and searching
the KDB.AI vector database for relevant document chunks.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

from src.embedding.embedder import get_embedder
from src.kdbai.client import get_client

logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("config/config.yaml")) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# Module-level embedder instance (lazy loaded)
_embedder = None


def get_query_embedder():
    """Get or create the query embedder instance.

    Uses the same FastEmbed model as the embedding pipeline to ensure
    query embeddings are in the same vector space as document embeddings.

    Returns:
        Embedder instance.
    """
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def embed_query(text: str) -> list[float]:
    """Generate an embedding for a user query.

    Uses the same FastEmbed model (BAAI/bge-small-en-v1.5) as the document
    embedding pipeline to ensure queries are in the same vector space.

    Args:
        text: The query text to embed.

    Returns:
        A list of floats representing the embedding vector (384 dimensions).

    Example:
        >>> embedding = embed_query("How do I create a table in q?")
        >>> len(embedding)
        384
    """
    if not text or not text.strip():
        raise ValueError("Query text cannot be empty")

    embedder = get_query_embedder()
    embedding = embedder.generate_embedding(text)

    # Convert numpy array to list for JSON serialization
    embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

    logger.debug(f"Generated embedding for query: '{text[:50]}...' ({len(embedding_list)} dims)")
    return embedding_list


def search_similar(
    query_text: str,
    top_k: int = 5,
    chapter_filter: Optional[str] = None,
) -> list[dict]:
    """Search for document chunks similar to the query.

    Args:
        query_text: The query text to search for.
        top_k: Number of results to return (default 5).
        chapter_filter: Optional chapter name to filter results.

    Returns:
        List of dicts with matching chunks and metadata:
        - chunk_id: Unique identifier
        - text: The chunk text content
        - chapter: Chapter name
        - heading: Section heading
        - url: Source URL
        - score: Similarity score (higher is more similar)
    """
    # Generate embedding for the query
    query_embedding = embed_query(query_text)

    # Build filter expression if chapter specified
    filter_expr = None
    if chapter_filter:
        filter_expr = [("=", "chapter", chapter_filter)]
        logger.debug(f"Applying chapter filter: {chapter_filter}")

    # Search KDB.AI
    client = get_client()
    results = client.search(
        query_vector=query_embedding,
        n=top_k,
        filter_expr=filter_expr,
    )

    logger.info(f"Found {len(results)} results for query: '{query_text[:50]}...'")
    return results


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.DEBUG)

    test_query = "How do I create a table in q?"
    print(f"Testing query embedding for: '{test_query}'")

    embedding = embed_query(test_query)
    print(f"Embedding shape: {len(embedding)} dimensions")
    print(f"First 5 values: {embedding[:5]}")
