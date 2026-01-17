"""
Heading-aware text chunker for markdown documents.

This module provides functionality to split markdown documents into chunks
based on heading hierarchy (H1/H2/H3), preserving context about the
document structure for each chunk. Large chunks are split with configurable
overlap to maintain context across chunk boundaries.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Default chunking parameters (can be overridden via config)
DEFAULT_CHUNK_SIZE = 512  # tokens
DEFAULT_CHUNK_OVERLAP = 50  # tokens


@dataclass
class Chunk:
    """A chunk of text with heading hierarchy metadata."""

    text: str
    heading_h1: str = ""
    heading_h2: str = ""
    heading_h3: str = ""
    chunk_index: int = 0
    source_file: str = ""

    @property
    def heading_hierarchy(self) -> str:
        """Return the full heading hierarchy as a path-like string."""
        parts = [h for h in [self.heading_h1, self.heading_h2, self.heading_h3] if h]
        return " > ".join(parts) if parts else ""

    @property
    def heading(self) -> str:
        """Return the most specific (deepest) heading."""
        for h in [self.heading_h3, self.heading_h2, self.heading_h1]:
            if h:
                return h
        return ""


def _parse_headings(line: str) -> tuple[int, str] | None:
    """
    Parse a markdown heading line.

    Args:
        line: A line of text that may be a heading.

    Returns:
        Tuple of (level, heading_text) if the line is a heading, None otherwise.
        Level 1 = H1, Level 2 = H2, Level 3 = H3.
    """
    # Match ATX-style headings (# Heading)
    match = re.match(r'^(#{1,3})\s+(.+)$', line.strip())
    if match:
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        return (level, heading_text)
    return None


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.

    Uses a simple heuristic: ~4 characters per token on average.
    This is a common approximation for English text and works well
    for embedding models like BAAI/bge-small-en-v1.5.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated number of tokens.
    """
    if not text:
        return 0
    # Approximate: 1 token ≈ 4 characters for English text
    return len(text) // 4


def split_text_with_overlap(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into chunks with overlap, respecting sentence/paragraph boundaries.

    Args:
        text: The text to split.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Number of tokens to overlap between chunks.

    Returns:
        List of text chunks with overlap.
    """
    if not text or not text.strip():
        return []

    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return [text]

    # Split into paragraphs first (double newline)
    paragraphs = re.split(r'\n\n+', text)
    if len(paragraphs) == 1:
        # If no paragraphs, split by sentences
        paragraphs = re.split(r'(?<=[.!?])\s+', text)
    if len(paragraphs) == 1:
        # If still one chunk, split by lines
        paragraphs = text.split('\n')

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # If single paragraph exceeds limit, split by words
        if para_tokens > max_tokens and not current_parts:
            words = para.split()
            word_chunk: list[str] = []
            word_tokens = 0

            for word in words:
                word_token_count = estimate_tokens(word + " ")
                if word_tokens + word_token_count > max_tokens and word_chunk:
                    chunks.append(' '.join(word_chunk))
                    # Overlap: keep last N tokens worth of words
                    overlap_words = []
                    overlap_count = 0
                    for w in reversed(word_chunk):
                        w_tokens = estimate_tokens(w + " ")
                        if overlap_count + w_tokens <= overlap_tokens:
                            overlap_words.insert(0, w)
                            overlap_count += w_tokens
                        else:
                            break
                    word_chunk = overlap_words
                    word_tokens = overlap_count

                word_chunk.append(word)
                word_tokens += word_token_count

            if word_chunk:
                current_parts = [' '.join(word_chunk)]
                current_tokens = estimate_tokens(current_parts[0])
            continue

        # Check if adding this paragraph exceeds limit
        if current_tokens + para_tokens > max_tokens and current_parts:
            # Save current chunk
            chunks.append('\n\n'.join(current_parts))

            # Create overlap from end of current chunk
            overlap_parts: list[str] = []
            overlap_count = 0
            for part in reversed(current_parts):
                part_tokens = estimate_tokens(part)
                if overlap_count + part_tokens <= overlap_tokens:
                    overlap_parts.insert(0, part)
                    overlap_count += part_tokens
                else:
                    break

            current_parts = overlap_parts
            current_tokens = overlap_count

        current_parts.append(para)
        current_tokens += para_tokens

    # Don't forget the last chunk
    if current_parts:
        chunks.append('\n\n'.join(current_parts))

    logger.debug(f"Split text into {len(chunks)} chunks with overlap")
    return chunks


def chunk_by_headings(
    markdown: str,
    source_file: str = "",
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    """
    Split markdown content into chunks based on H1/H2/H3 headings.

    Each chunk contains the text under a heading section, along with
    metadata about the heading hierarchy (H1 > H2 > H3). Large sections
    are automatically split with configurable overlap to maintain context.

    Args:
        markdown: The markdown content to chunk.
        source_file: Optional source filename for metadata.
        max_tokens: Maximum tokens per chunk. If None, uses config or default (512).
        overlap_tokens: Token overlap between chunks. If None, uses config or default (50).

    Returns:
        List of Chunk objects, each with text and heading hierarchy metadata.

    Example:
        >>> md = '''
        ... # Chapter 1
        ... Introduction text.
        ... ## Section 1.1
        ... Section content here.
        ... ### Subsection 1.1.1
        ... Detailed content.
        ... '''
        >>> chunks = chunk_by_headings(md)
        >>> len(chunks)
        3
        >>> chunks[0].heading_h1
        'Chapter 1'
        >>> chunks[1].heading_hierarchy
        'Chapter 1 > Section 1.1'
    """
    if not markdown or not markdown.strip():
        logger.warning("Empty markdown content provided")
        return []

    # Load config values if parameters not provided
    config = load_config()
    embedding_config = config.get("embedding", {})

    if max_tokens is None:
        max_tokens = embedding_config.get("chunk_size", DEFAULT_CHUNK_SIZE)
    if overlap_tokens is None:
        overlap_tokens = embedding_config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)

    logger.debug(f"Chunking with max_tokens={max_tokens}, overlap_tokens={overlap_tokens}")

    lines = markdown.split('\n')
    heading_chunks: list[tuple[str, str, str, str]] = []  # (text, h1, h2, h3)

    # Current heading context
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""

    # Current chunk content
    current_lines: list[str] = []

    def _save_heading_chunk():
        """Save the current accumulated lines as a heading-based chunk."""
        text = '\n'.join(current_lines).strip()
        if text:
            heading_chunks.append((text, current_h1, current_h2, current_h3))

    for line in lines:
        heading_info = _parse_headings(line)

        if heading_info:
            level, heading_text = heading_info

            # Save current chunk before starting new section
            _save_heading_chunk()
            current_lines = []

            # Update heading hierarchy based on level
            if level == 1:
                current_h1 = heading_text
                current_h2 = ""  # Reset lower levels
                current_h3 = ""
            elif level == 2:
                current_h2 = heading_text
                current_h3 = ""  # Reset lower level
            elif level == 3:
                current_h3 = heading_text

            # Include the heading itself in the chunk
            current_lines.append(line)
        else:
            # Regular content line
            current_lines.append(line)

    # Don't forget the last chunk
    _save_heading_chunk()

    # Now split oversized chunks and create final Chunk objects
    chunks: list[Chunk] = []
    chunk_index = 0

    for text, h1, h2, h3 in heading_chunks:
        # Split if chunk exceeds token limit
        split_texts = split_text_with_overlap(text, max_tokens, overlap_tokens)

        for split_text in split_texts:
            chunk = Chunk(
                text=split_text,
                heading_h1=h1,
                heading_h2=h2,
                heading_h3=h3,
                chunk_index=chunk_index,
                source_file=source_file,
            )
            chunks.append(chunk)
            logger.debug(
                f"Created chunk {chunk_index}: {chunk.heading_hierarchy or '(no heading)'} "
                f"({len(split_text)} chars, ~{estimate_tokens(split_text)} tokens)"
            )
            chunk_index += 1

    logger.info(
        f"Created {len(chunks)} chunks from {len(lines)} lines "
        f"({len(heading_chunks)} heading sections)"
    )
    return chunks


def load_config() -> dict:
    """Load configuration from config/config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# Module test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test estimate_tokens
    print("Testing estimate_tokens()...")
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") == 2  # 11 chars / 4 ≈ 2
    assert estimate_tokens("a" * 100) == 25  # 100 chars / 4 = 25
    print("  ✓ estimate_tokens works correctly")

    # Test split_text_with_overlap with small limit
    print("\nTesting split_text_with_overlap()...")
    short_text = "This is short."
    splits = split_text_with_overlap(short_text, max_tokens=100, overlap_tokens=10)
    assert len(splits) == 1
    assert splits[0] == short_text
    print("  ✓ Short text not split")

    # Test with long text that needs splitting
    long_text = " ".join(["word"] * 500)  # ~2000 chars = ~500 tokens
    splits = split_text_with_overlap(long_text, max_tokens=100, overlap_tokens=20)
    assert len(splits) > 1
    print(f"  ✓ Long text split into {len(splits)} chunks")

    # Verify overlap exists (chunks should share some content)
    if len(splits) > 1:
        chunk1_words = set(splits[0].split()[-20:])  # Last 20 words of first chunk
        chunk2_words = set(splits[1].split()[:20])   # First 20 words of second chunk
        overlap = chunk1_words & chunk2_words
        assert len(overlap) > 0, "Chunks should have overlap"
        print(f"  ✓ Overlap verified ({len(overlap)} words shared)")

    # Test markdown
    test_md = """# Chapter 1: Introduction

This is the introduction to the chapter. It explains the basics.

## Section 1.1: Getting Started

Here we cover how to get started with the topic.

Some more content in this section.

### Subsection 1.1.1: First Steps

These are the first steps you should take.

### Subsection 1.1.2: Next Steps

After completing first steps, do these.

## Section 1.2: Advanced Topics

This section covers more advanced topics.

# Chapter 2: Deep Dive

Now we go deeper into the subject.

## Section 2.1: Core Concepts

Understanding the core concepts is essential.
"""

    print("\nTesting chunk_by_headings() without size limits...")
    chunks = chunk_by_headings(test_md, source_file="test.md", max_tokens=10000)

    print(f"\nCreated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        tokens = estimate_tokens(chunk.text)
        print(f"--- Chunk {i} ({tokens} tokens) ---")
        print(f"Hierarchy: {chunk.heading_hierarchy}")
        print(f"Text preview: {chunk.text[:80]}...")
        print()

    # Test with very small chunk size to force splitting
    print("\nTesting chunk_by_headings() with small chunk size (50 tokens)...")
    small_chunks = chunk_by_headings(test_md, source_file="test.md", max_tokens=50, overlap_tokens=10)
    print(f"Created {len(small_chunks)} chunks (vs {len(chunks)} without limits)")
    assert len(small_chunks) >= len(chunks), "Should have at least as many chunks when limiting size"

    for i, chunk in enumerate(small_chunks):
        tokens = estimate_tokens(chunk.text)
        print(f"  Chunk {i}: {tokens} tokens - {chunk.heading or '(no heading)'}")

    print("\n✓ All tests passed!")
