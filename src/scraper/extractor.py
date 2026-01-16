"""Content extraction from HTML pages.

This module provides functions to extract clean body content from HTML,
removing navigation, TOC, headers, footers, and other non-content elements.
"""

import logging
from typing import List, Optional

from bs4 import BeautifulSoup

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


if __name__ == "__main__":
    # Test the extractor with sample HTML
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
                <p>More content here.</p>
            </article>
        </main>
        <footer>Footer</footer>
    </body>
    </html>
    """

    result = extract_body(sample_html)
    print("Extracted content:")
    print(result)
