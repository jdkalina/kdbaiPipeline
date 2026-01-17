"""Main scraper orchestrator.

This module orchestrates the full scrape pipeline:
1. Discover chapter URLs
2. Fetch each page with rate limiting
3. Extract and convert content to Markdown
4. Save to disk
5. Update SQLite database with status

Includes idempotency support: content hashes are checked on re-scrape
to skip files that haven't changed.

Usage:
    python -m src.scraper.main [--config CONFIG_PATH]
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml

from src.db import init_db, insert_scraped_file, get_file_by_url, update_content_hash
from src.scraper.discover import discover_chapter_urls
from src.scraper.extractor import extract_and_convert
from src.scraper.fetcher import Fetcher
from src.scraper.storage import save_content, compute_hash, url_to_filename

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_PATH = Path("config/config.yaml")
DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_DB_PATH = Path("data/status.db")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    """Configure logging based on config settings.

    Args:
        config: Configuration dictionary.
    """
    log_config = config.get("logging", {})
    level = log_config.get("level", "INFO")
    format_str = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logging.basicConfig(level=level, format=format_str)


def run_scraper(
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> dict:
    """Run the full scrape pipeline.

    Discovers URLs, fetches pages, extracts content, saves to disk, and updates
    the database. Includes progress logging for each step.

    Idempotency: Content hashes are compared on re-scrape. If a URL already
    exists in the database and its content hash matches, the file is skipped.
    If the hash differs, the file is updated.

    Args:
        config_path: Path to config file. Defaults to config/config.yaml.
        output_dir: Directory to save scraped files. Defaults to data/raw/.
        db_path: Path to SQLite database. Defaults to data/status.db.

    Returns:
        Dictionary with scrape statistics:
        - total: Total URLs discovered
        - scraped: Number of new files scraped
        - updated: Number of files updated (content changed)
        - skipped: Number skipped (content unchanged)
        - failed: Number that failed
    """
    # Load configuration
    config_path = config_path or DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    setup_logging(config)

    # Extract settings
    scraper_config = config.get("scraper", {})
    base_url = scraper_config.get("base_url", "https://code.kx.com/q4m3/")
    delay = scraper_config.get("delay", 1.0)
    timeout = scraper_config.get("timeout", 30)

    # Set up paths
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = db_path or Path(config.get("database", {}).get("path", DEFAULT_DB_PATH))
    db_path = Path(db_path)

    logger.info("=" * 60)
    logger.info("Q for Mortals Scraper - Starting")
    logger.info("=" * 60)
    logger.info(f"Base URL: {base_url}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Database path: {db_path}")
    logger.info(f"Rate limit delay: {delay}s")

    # Initialize database
    conn = init_db(db_path)

    # Statistics
    stats = {
        "total": 0,
        "scraped": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
    }

    # Discover URLs
    logger.info("-" * 60)
    logger.info("Phase 1: Discovering chapter URLs")
    logger.info("-" * 60)

    try:
        urls = discover_chapter_urls(base_url, timeout)
        stats["total"] = len(urls)
        logger.info(f"Discovered {len(urls)} chapter URLs")
    except Exception as e:
        logger.error(f"Failed to discover URLs: {e}")
        conn.close()
        return stats

    # Scrape each URL
    logger.info("-" * 60)
    logger.info("Phase 2: Fetching and processing pages")
    logger.info("-" * 60)

    with Fetcher(delay=delay, timeout=timeout) as fetcher:
        for i, url in enumerate(urls, 1):
            logger.info(f"[{i}/{len(urls)}] Processing: {url}")

            # Check if already in database
            existing = get_file_by_url(conn, url)

            try:
                # Fetch the page
                html = fetcher.fetch(url)

                # Extract and convert to markdown
                markdown = extract_and_convert(html)

                if not markdown or not markdown.strip():
                    logger.warning(f"  -> No content extracted from {url}")
                    stats["failed"] += 1
                    continue

                # Compute content hash
                content_hash = compute_hash(markdown)

                # Check idempotency: if file exists and hash matches, skip
                if existing:
                    old_hash = existing["content_hash"]
                    if old_hash == content_hash:
                        logger.info(f"  -> Skipping (content unchanged, hash: {content_hash[:12]}...)")
                        stats["skipped"] += 1
                        continue
                    else:
                        # Content has changed - update the file
                        logger.info(f"  -> Content changed (old: {old_hash[:12]}..., new: {content_hash[:12]}...)")
                        filename = save_content(url, markdown, output_dir)
                        update_content_hash(conn, existing["file_id"], content_hash)
                        logger.info(f"  -> Updated: {filename} ({len(markdown)} chars)")
                        stats["updated"] += 1
                        continue

                # New file - save to disk and insert into database
                filename = save_content(url, markdown, output_dir)
                insert_scraped_file(conn, url, filename, content_hash, status="scraped")
                logger.info(f"  -> Saved: {filename} ({len(markdown)} chars)")
                stats["scraped"] += 1

            except Exception as e:
                logger.error(f"  -> Failed: {e}")
                stats["failed"] += 1

    # Close database connection
    conn.close()

    # Summary
    logger.info("-" * 60)
    logger.info("Scrape Complete - Summary")
    logger.info("-" * 60)
    logger.info(f"Total URLs:  {stats['total']}")
    logger.info(f"New:         {stats['scraped']}")
    logger.info(f"Updated:     {stats['updated']}")
    logger.info(f"Unchanged:   {stats['skipped']}")
    logger.info(f"Failed:      {stats['failed']}")
    logger.info("=" * 60)

    return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape Q for Mortals documentation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save scraped files (default: data/raw/)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to SQLite database (default: data/status.db)",
    )

    args = parser.parse_args()

    stats = run_scraper(
        config_path=args.config,
        output_dir=args.output_dir,
        db_path=args.db_path,
    )

    # Exit with error code if any failures
    if stats["failed"] > 0:
        exit(1)


if __name__ == "__main__":
    main()
