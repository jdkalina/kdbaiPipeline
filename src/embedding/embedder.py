"""FastEmbed wrapper for generating text embeddings.

This module provides a simple interface to the FastEmbed library for
generating embeddings using the BAAI/bge-small-en-v1.5 model.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from fastembed import TextEmbedding

import yaml


logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("config/config.yaml")) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class Embedder:
    """Wrapper for FastEmbed text embedding model.

    Provides methods for generating embeddings from text using the
    BAAI/bge-small-en-v1.5 model (384 dimensions).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        config_path: Path = Path("config/config.yaml")
    ):
        """Initialize the embedder.

        Args:
            model_name: Name of the FastEmbed model to use.
                       Defaults to config value.
            batch_size: Batch size for embedding generation.
                       Defaults to config value.
            config_path: Path to configuration file.
        """
        config = load_config(config_path)
        embedding_config = config.get("embedding", {})

        self.model_name = model_name or embedding_config.get(
            "model_name", "BAAI/bge-small-en-v1.5"
        )
        self.batch_size = batch_size or embedding_config.get("batch_size", 32)
        self.dimensions = embedding_config.get("dimensions", 384)

        logger.info(f"Initializing FastEmbed model: {self.model_name}")
        self._model: Optional[TextEmbedding] = None

    @property
    def model(self) -> TextEmbedding:
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info(f"Loading model {self.model_name}...")
            self._model = TextEmbedding(self.model_name)
            logger.info("Model loaded successfully")
        return self._model

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[np.ndarray]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            List of numpy arrays, each of shape (dimensions,).
        """
        if not texts:
            logger.warning("Empty text list provided, returning empty list")
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts")

        # FastEmbed returns a generator, convert to list
        embeddings = list(self.model.embed(
            texts,
            batch_size=self.batch_size,
        ))

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Numpy array of shape (dimensions,).
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0]


# Convenience functions for simple usage
_embedder: Optional[Embedder] = None


def get_embedder(
    model_name: Optional[str] = None,
    config_path: Path = Path("config/config.yaml")
) -> Embedder:
    """Get or create a singleton Embedder instance.

    Args:
        model_name: Optional model name override.
        config_path: Path to configuration file.

    Returns:
        Embedder instance.
    """
    global _embedder
    if _embedder is None:
        _embedder = Embedder(model_name=model_name, config_path=config_path)
    return _embedder


def generate_embeddings(
    texts: List[str],
    model_name: Optional[str] = None,
    config_path: Path = Path("config/config.yaml")
) -> List[np.ndarray]:
    """Generate embeddings for a list of texts using default embedder.

    Args:
        texts: List of text strings to embed.
        model_name: Optional model name override.
        config_path: Path to configuration file.

    Returns:
        List of numpy arrays, each of shape (dimensions,).
    """
    embedder = get_embedder(model_name=model_name, config_path=config_path)
    return embedder.generate_embeddings(texts)


def generate_embedding(
    text: str,
    model_name: Optional[str] = None,
    config_path: Path = Path("config/config.yaml")
) -> np.ndarray:
    """Generate embedding for a single text using default embedder.

    Args:
        text: Text string to embed.
        model_name: Optional model name override.
        config_path: Path to configuration file.

    Returns:
        Numpy array of shape (dimensions,).
    """
    embedder = get_embedder(model_name=model_name, config_path=config_path)
    return embedder.generate_embedding(text)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    test_texts = [
        "What is a q table?",
        "How do I create a dictionary in q?",
        "The select statement is used for querying tables."
    ]

    print("Testing FastEmbed wrapper...")
    embedder = Embedder()
    embeddings = embedder.generate_embeddings(test_texts)

    print(f"\nGenerated {len(embeddings)} embeddings:")
    for i, (text, emb) in enumerate(zip(test_texts, embeddings)):
        print(f"  {i+1}. '{text[:40]}...' -> shape={emb.shape}, dtype={emb.dtype}")

    # Test single embedding
    single_emb = embedder.generate_embedding("test query")
    print(f"\nSingle embedding shape: {single_emb.shape}")
    print(f"First 5 values: {single_emb[:5]}")
