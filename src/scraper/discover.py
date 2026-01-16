"""URL discovery for Q for Mortals chapters.

This module fetches the Q4M3 index page and extracts all chapter URLs.
"""

import logging
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Known chapter URL patterns based on Q4M3 structure
CHAPTER_PATTERNS = [
    "preface/",
    "0_Overview/",
    "1_Q_Shock_and_Awe/",
    "2_Basic_Data_Types_Atoms/",
    "3_Lists/",
    "4_Operators/",
    "5_Dictionaries/",
    "6_Functions/",
    "7_Transforming_Data/",
    "8_Tables/",
    "9_Queries_q-sql/",
    "10_Execution_Control/",
    "11_IO/",
    "12_Workspace_Organization/",
    "13_Commands_and_System_Variables/",
    "14_Introduction_to_Kdb+/",
    "A_Built-in_Functions/",
    "B_Error_Messages/",
    "colophon/",
]


def discover_chapter_urls(base_url: str = "https://code.kx.com/q4m3/", timeout: int = 30) -> list[str]:
    """Discover all chapter URLs from the Q for Mortals index page.

    Args:
        base_url: The base URL of the Q4M3 documentation.
        timeout: Request timeout in seconds.

    Returns:
        A list of absolute URLs for each chapter (main pages only, no section anchors).

    Raises:
        requests.RequestException: If the HTTP request fails.
    """
    logger.info(f"Discovering chapter URLs from {base_url}")

    response = requests.get(base_url, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    chapter_urls: list[str] = []

    # Find all links and match against known patterns
    for link in soup.find_all("a", href=True):
        href = link["href"]

        # Skip anchor links (section links within a page)
        if "#" in href:
            continue

        # Check if href matches any known chapter pattern
        for pattern in CHAPTER_PATTERNS:
            if pattern in href or href.endswith(pattern) or href.endswith(pattern.rstrip("/")):
                absolute_url = urljoin(base_url, href)
                # Normalize URL to end with /
                if not absolute_url.endswith("/"):
                    absolute_url += "/"
                if absolute_url not in chapter_urls:
                    chapter_urls.append(absolute_url)
                break

    # If dynamic discovery didn't find all chapters, use the known list as fallback
    if len(chapter_urls) < len(CHAPTER_PATTERNS):
        logger.warning(
            f"Dynamic discovery found {len(chapter_urls)} chapters, "
            f"expected {len(CHAPTER_PATTERNS)}. Using known chapter list."
        )
        chapter_urls = [urljoin(base_url, pattern) for pattern in CHAPTER_PATTERNS]

    logger.info(f"Discovered {len(chapter_urls)} chapter URLs")
    return chapter_urls


if __name__ == "__main__":
    # Enable logging for testing
    logging.basicConfig(level=logging.INFO)

    urls = discover_chapter_urls()
    for url in urls:
        print(url)
