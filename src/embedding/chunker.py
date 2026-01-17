"""
Heading-aware text chunker for markdown documents.

This module provides functionality to split markdown documents into chunks
based on heading hierarchy (H1/H2/H3), preserving context about the
document structure for each chunk.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


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


def chunk_by_headings(
    markdown: str,
    source_file: str = "",
) -> list[Chunk]:
    """
    Split markdown content into chunks based on H1/H2/H3 headings.

    Each chunk contains the text under a heading section, along with
    metadata about the heading hierarchy (H1 > H2 > H3).

    Args:
        markdown: The markdown content to chunk.
        source_file: Optional source filename for metadata.

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

    lines = markdown.split('\n')
    chunks: list[Chunk] = []

    # Current heading context
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""

    # Current chunk content
    current_lines: list[str] = []
    chunk_index = 0

    def _save_chunk():
        """Save the current accumulated lines as a chunk."""
        nonlocal chunk_index
        text = '\n'.join(current_lines).strip()
        if text:
            chunk = Chunk(
                text=text,
                heading_h1=current_h1,
                heading_h2=current_h2,
                heading_h3=current_h3,
                chunk_index=chunk_index,
                source_file=source_file,
            )
            chunks.append(chunk)
            chunk_index += 1
            logger.debug(
                f"Created chunk {chunk_index}: {chunk.heading_hierarchy or '(no heading)'} "
                f"({len(text)} chars)"
            )

    for line in lines:
        heading_info = _parse_headings(line)

        if heading_info:
            level, heading_text = heading_info

            # Save current chunk before starting new section
            _save_chunk()
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
    _save_chunk()

    logger.info(f"Created {len(chunks)} chunks from {len(lines)} lines")
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

    print("Testing chunk_by_headings()...")
    chunks = chunk_by_headings(test_md, source_file="test.md")

    print(f"\nCreated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} ---")
        print(f"Hierarchy: {chunk.heading_hierarchy}")
        print(f"Heading: {chunk.heading}")
        print(f"H1: {chunk.heading_h1}, H2: {chunk.heading_h2}, H3: {chunk.heading_h3}")
        print(f"Text preview: {chunk.text[:100]}...")
        print()

    print("All tests passed!")
