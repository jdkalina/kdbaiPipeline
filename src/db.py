"""
SQLite database operations for status tracking.

This module provides functions to manage the SQLite database that tracks
the status of scraped files through the RAG pipeline.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def init_db(db_path: Path) -> sqlite3.Connection:
    """
    Initialize the SQLite database with the required schema.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection: Active database connection.
    """
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable dict-like access to rows

    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_id TEXT PRIMARY KEY,
            url TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            scraped_at TEXT,
            content_hash TEXT,
            status TEXT DEFAULT 'pending',
            error_message TEXT
        )
    """)

    # Create index on status for efficient filtering
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_files_status ON files(status)
    """)

    # Create index on url for efficient lookups
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_files_url ON files(url)
    """)

    conn.commit()
    logger.info(f"Database initialized at {db_path}")

    return conn
