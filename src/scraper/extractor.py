"""Content extraction from HTML pages.

This module provides functions to extract clean body content from HTML,
removing navigation, TOC, headers, footers, and other non-content elements.
"""

import logging
from typing import List, Optional

from bs4 import BeautifulSoup
import markdownify

logger = logging.getLogger(__name__)

# Default selectors for content extraction
DEFAULT_CONTENT_SELECTORS = [
    "main",
    "article",
    "div.md-content",
    "div.content",
]

# Default elements to remove
DEFAULT_REMOVE_SELECTORS = [
    "nav",
    "header",
    "footer",
    "aside",
    "div.md-sidebar",
    "div.toc",
    "div.admonition.todo",
    "script",
    "style",
    "noscript",
]


def extract_body(
    html: str,
    content_selectors: Optional[List[str]] = None,
    remove_selectors: Optional[List[str]] = None,
) -> str:
    """Extract clean body content from HTML.

    Removes navigation, TOC, headers, footers, and other non-content elements.
    Returns the main content area as clean HTML.

    Args:
        html: The raw HTML content.
        content_selectors: CSS selectors to try for finding main content
            (tried in order, first match wins). Defaults to common selectors.
        remove_selectors: CSS selectors for elements to remove from content.
            Defaults to common non-content elements.

    Returns:
        Clean HTML string containing only the main content.
        Returns empty string if no content is found.
    """
    if content_selectors is None:
        content_selectors = DEFAULT_CONTENT_SELECTORS
    if remove_selectors is None:
        remove_selectors = DEFAULT_REMOVE_SELECTORS

    soup = BeautifulSoup(html, "html.parser")

    # Find the main content area
    content = None
    for selector in content_selectors:
        content = soup.select_one(selector)
        if content:
            logger.debug(f"Found content using selector: {selector}")
            break

    # Fall back to body if no main content area found
    if content is None:
        content = soup.body
        if content:
            logger.debug("Falling back to <body> element")
        else:
            logger.warning("No body element found in HTML")
            return ""

    # Remove unwanted elements
    for selector in remove_selectors:
        for element in content.select(selector):
            logger.debug(f"Removing element matching: {selector}")
            element.decompose()

    # Get the cleaned HTML
    cleaned_html = str(content)

    logger.info(
        f"Extracted {len(cleaned_html)} characters of content "
        f"(original: {len(html)} characters)"
    )

    return cleaned_html


def convert_to_markdown(
    html: str,
    code_language: str = "q",
    heading_style: str = "ATX",
) -> str:
    """Convert HTML to well-formatted Markdown.

    Converts clean HTML content to Markdown format, preserving code blocks,
    headings, lists, and other structural elements.

    Args:
        html: The HTML content to convert.
        code_language: Default language to use for code blocks (default: "q"
            for Q/kdb+ code).
        heading_style: Style for headings - "ATX" uses # symbols, "SETEXT"
            uses underlines (default: "ATX").

    Returns:
        Markdown-formatted string.
    """
    if not html or not html.strip():
        logger.warning("Empty HTML provided for markdown conversion")
        return ""

    # Configure markdownify options
    # - heading_style: ATX uses # symbols
    # - code_language: default language for code blocks
    # - strip: remove these tags entirely
    # - bullets: bullet character for unordered lists
    markdown = markdownify.markdownify(
        html,
        heading_style=heading_style,
        code_language=code_language,
        bullets="-",
        strip=["script", "style", "noscript"],
    )

    # Clean up the markdown
    # Remove excessive blank lines (more than 2 consecutive)
    lines = markdown.split("\n")
    cleaned_lines = []
    blank_count = 0

    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    markdown = "\n".join(cleaned_lines)

    # Strip leading/trailing whitespace
    markdown = markdown.strip()

    logger.info(
        f"Converted HTML to Markdown: {len(html)} chars -> {len(markdown)} chars"
    )

    return markdown


def extract_and_convert(
    html: str,
    content_selectors: Optional[List[str]] = None,
    remove_selectors: Optional[List[str]] = None,
    code_language: str = "q",
) -> str:
    """Extract content from HTML and convert to Markdown in one step.

    Convenience function that combines extract_body() and convert_to_markdown().

    Args:
        html: The raw HTML content.
        content_selectors: CSS selectors to try for finding main content.
        remove_selectors: CSS selectors for elements to remove from content.
        code_language: Default language for code blocks.

    Returns:
        Markdown-formatted string of the main content.
    """
    clean_html = extract_body(html, content_selectors, remove_selectors)
    if not clean_html:
        return ""
    return convert_to_markdown(clean_html, code_language=code_language)


if __name__ == "__main__":
    # Test the extractor and markdown conversion
    logging.basicConfig(level=logging.DEBUG)

    sample_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test</title></head>
    <body>
        <header><nav>Navigation</nav></header>
        <aside class="md-sidebar">Sidebar</aside>
        <main>
            <article>
                <h1>Main Content</h1>
                <p>This is the main content.</p>
                <div class="toc">Table of Contents</div>
                <h2>Code Example</h2>
                <pre><code>q) 2+2
4
q) til 10
0 1 2 3 4 5 6 7 8 9</code></pre>
                <p>More content here with <strong>bold</strong> and <em>italic</em> text.</p>
                <ul>
                    <li>First item</li>
                    <li>Second item</li>
                </ul>
            </article>
        </main>
        <footer>Footer</footer>
    </body>
    </html>
    """

    print("=" * 60)
    print("Testing extract_body():")
    print("=" * 60)
    extracted = extract_body(sample_html)
    print(extracted)

    print("\n" + "=" * 60)
    print("Testing convert_to_markdown():")
    print("=" * 60)
    markdown = convert_to_markdown(extracted)
    print(markdown)

    print("\n" + "=" * 60)
    print("Testing extract_and_convert():")
    print("=" * 60)
    combined = extract_and_convert(sample_html)
    print(combined)
