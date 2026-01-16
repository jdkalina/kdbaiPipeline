"""HTML fetcher with rate limiting for polite web scraping.

This module provides HTTP fetching functionality with configurable delay
between requests to avoid overwhelming the target server.
"""

import logging
import time
from typing import Optional

import requests
from requests.exceptions import HTTPError, RequestException, Timeout

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_DELAY = 1.0  # seconds between requests
DEFAULT_TIMEOUT = 30  # seconds


class Fetcher:
    """HTTP fetcher with rate limiting support.

    Maintains state for rate limiting across multiple requests.
    """

    def __init__(
        self,
        delay: float = DEFAULT_DELAY,
        timeout: int = DEFAULT_TIMEOUT,
        user_agent: Optional[str] = None,
    ):
        """Initialize the fetcher.

        Args:
            delay: Seconds to wait between requests.
            timeout: Request timeout in seconds.
            user_agent: Optional custom User-Agent header.
        """
        self.delay = delay
        self.timeout = timeout
        self._last_request_time: Optional[float] = None

        self.session = requests.Session()
        if user_agent:
            self.session.headers["User-Agent"] = user_agent
        else:
            # Use a descriptive user agent
            self.session.headers["User-Agent"] = (
                "Q4M-RAG-Agent/1.0 (Educational scraper for Q for Mortals documentation)"
            )

    def _wait_for_rate_limit(self) -> None:
        """Wait if needed to respect rate limiting."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.delay:
                wait_time = self.delay - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                time.sleep(wait_time)

    def fetch(self, url: str) -> str:
        """Fetch a page with rate limiting.

        Args:
            url: The URL to fetch.

        Returns:
            The HTML content of the page.

        Raises:
            requests.HTTPError: If the server returns an error status code.
            requests.Timeout: If the request times out.
            requests.RequestException: For other request errors.
        """
        self._wait_for_rate_limit()

        logger.info(f"Fetching: {url}")

        try:
            response = self.session.get(url, timeout=self.timeout)
            self._last_request_time = time.time()

            response.raise_for_status()

            logger.debug(f"Fetched {len(response.text)} bytes from {url}")
            return response.text

        except Timeout:
            logger.error(f"Timeout fetching {url}")
            raise
        except HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code} fetching {url}")
            raise
        except RequestException as e:
            logger.error(f"Request error fetching {url}: {e}")
            raise

    def close(self) -> None:
        """Close the underlying session."""
        self.session.close()

    def __enter__(self) -> "Fetcher":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def fetch_page(
    url: str,
    delay: float = DEFAULT_DELAY,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Fetch a single page with rate limiting.

    This is a convenience function for one-off fetches. For multiple requests,
    use the Fetcher class to maintain rate limiting state across requests.

    Args:
        url: The URL to fetch.
        delay: Seconds to wait before the request (applied immediately).
        timeout: Request timeout in seconds.

    Returns:
        The HTML content of the page.

    Raises:
        requests.HTTPError: If the server returns an error status code.
        requests.Timeout: If the request times out.
        requests.RequestException: For other request errors.
    """
    # Apply delay before making the request
    if delay > 0:
        logger.debug(f"Waiting {delay}s before request")
        time.sleep(delay)

    logger.info(f"Fetching: {url}")

    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "Q4M-RAG-Agent/1.0 (Educational scraper for Q for Mortals documentation)"
        },
    )
    response.raise_for_status()

    logger.debug(f"Fetched {len(response.text)} bytes")
    return response.text


if __name__ == "__main__":
    # Test the fetcher
    logging.basicConfig(level=logging.DEBUG)

    # Test with a single fetch
    html = fetch_page("https://code.kx.com/q4m3/", delay=0)
    print(f"Fetched {len(html)} bytes")

    # Test with the Fetcher class for multiple requests
    with Fetcher(delay=0.5) as fetcher:
        html1 = fetcher.fetch("https://code.kx.com/q4m3/")
        html2 = fetcher.fetch("https://code.kx.com/q4m3/preface/")
        print(f"Fetched {len(html1)} and {len(html2)} bytes")
