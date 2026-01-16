"""File storage for scraped content.

This module provides functions to save scraped content to disk with unique
filenames derived from URLs, and to check for existing files.
"""

import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_OUTPUT_DIR = Path("data/raw")


def url_to_filename(url: str) -> str:
    """Generate a slug-based filename from a URL.

    Extracts the path component of the URL and converts it to a safe filename
    suitable for filesystem storage.

    Args:
        url: The URL to convert to a filename.

    Returns:
        A filename string ending in .md (e.g., "1_q_sql.md").
    """
    parsed = urlparse(url)
    # Decode URL-encoded characters (e.g., %2B -> +)
    path = unquote(parsed.path)

    # Remove leading/trailing slashes and the base path if present
    path = path.strip("/")

    # Remove common base paths (e.g., "q4m3/")
    if "/" in path:
        # Take the last meaningful segment
        segments = [s for s in path.split("/") if s]
        if len(segments) > 1:
            # Remove the base path (e.g., "q4m3")
            path = "_".join(segments[1:])
        else:
            path = segments[0] if segments else "index"
    elif not path:
        path = "index"

    # Clean up the path to make it filename-safe
    # Replace + with 'plus' to preserve meaning (e.g., kdb+ -> kdb_plus)
    path = path.replace("+", "_plus")

    # Replace other URL-unsafe chars with underscores
    filename = re.sub(r"[^a-zA-Z0-9_-]", "_", path)

    # Collapse multiple underscores
    filename = re.sub(r"_+", "_", filename)

    # Remove leading/trailing underscores
    filename = filename.strip("_")

    # Ensure we have a filename
    if not filename:
        filename = "index"

    # Add .md extension
    filename = f"{filename}.md"

    logger.debug(f"Converted URL '{url}' to filename '{filename}'")
    return filename


def save_content(
    url: str,
    content: str,
    output_dir: Optional[Path] = None,
) -> str:
    """Save scraped content to a file with a slug-based filename.

    Creates the output directory if it doesn't exist. The filename is
    generated from the URL to ensure uniqueness and readability.

    Args:
        url: The source URL (used to generate the filename).
        content: The content to save (typically Markdown).
        output_dir: Directory to save files to. Defaults to data/raw/.

    Returns:
        The filename (not full path) of the saved file.

    Raises:
        OSError: If the file cannot be written.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from URL
    filename = url_to_filename(url)
    file_path = output_dir / filename

    # Write content to file
    file_path.write_text(content, encoding="utf-8")

    logger.info(f"Saved content to {file_path} ({len(content)} characters)")
    return filename


def file_exists(
    url: str,
    output_dir: Optional[Path] = None,
) -> bool:
    """Check if a file already exists for a given URL.

    Args:
        url: The source URL.
        output_dir: Directory to check in. Defaults to data/raw/.

    Returns:
        True if the file exists, False otherwise.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)
    filename = url_to_filename(url)
    file_path = output_dir / filename

    exists = file_path.exists()
    logger.debug(f"File exists check for {url}: {exists}")
    return exists


def get_file_path(
    url: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Get the full file path for a given URL.

    Args:
        url: The source URL.
        output_dir: Directory where files are stored. Defaults to data/raw/.

    Returns:
        Full Path object to the file location.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)
    filename = url_to_filename(url)
    return output_dir / filename


def read_content(
    url: str,
    output_dir: Optional[Path] = None,
) -> Optional[str]:
    """Read content from a file for a given URL.

    Args:
        url: The source URL.
        output_dir: Directory where files are stored. Defaults to data/raw/.

    Returns:
        The file content as a string, or None if file doesn't exist.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    file_path = get_file_path(url, output_dir)

    if not file_path.exists():
        logger.debug(f"File not found for URL: {url}")
        return None

    content = file_path.read_text(encoding="utf-8")
    logger.debug(f"Read {len(content)} characters from {file_path}")
    return content


if __name__ == "__main__":
    # Test the storage module
    logging.basicConfig(level=logging.DEBUG)

    # Test URL to filename conversion
    test_urls = [
        "https://code.kx.com/q4m3/",
        "https://code.kx.com/q4m3/1_Q_Shock_and_Awe/",
        "https://code.kx.com/q4m3/2_Basic_Data_Types_Atoms/",
        "https://code.kx.com/q4m3/14_Introduction_to_Kdb%2B/",
        "https://code.kx.com/q4m3/A_Built-in_Functions/",
    ]

    print("=" * 60)
    print("Testing url_to_filename():")
    print("=" * 60)
    for url in test_urls:
        filename = url_to_filename(url)
        print(f"  {url}")
        print(f"    -> {filename}")
        print()

    # Test saving content
    print("=" * 60)
    print("Testing save_content():")
    print("=" * 60)
    test_content = "# Test Content\n\nThis is a test file.\n"
    test_url = "https://code.kx.com/q4m3/test_chapter/"

    # Use a temp directory for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        filename = save_content(test_url, test_content, output_dir)
        print(f"Saved to: {filename}")

        # Test file_exists
        print(f"File exists: {file_exists(test_url, output_dir)}")

        # Test read_content
        read_back = read_content(test_url, output_dir)
        print(f"Read back matches: {read_back == test_content}")
