"""
SQLite database operations for status tracking.

This module provides functions to manage the SQLite database that tracks
the status of scraped files through the RAG pipeline.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

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


def insert_scraped_file(
    conn: sqlite3.Connection,
    url: str,
    filename: str,
    content_hash: str,
    status: str = "scraped"
) -> str:
    """
    Insert a new scraped file record into the database.

    Args:
        conn: Active database connection.
        url: The source URL of the scraped content.
        filename: The local filename where content is saved.
        content_hash: SHA256 hash of the content.
        status: Status of the file (default: "scraped").

    Returns:
        str: The generated file_id.

    Raises:
        sqlite3.IntegrityError: If a file with this URL already exists.
    """
    file_id = str(uuid.uuid4())
    scraped_at = datetime.utcnow().isoformat()

    conn.execute(
        """
        INSERT INTO files (file_id, url, filename, scraped_at, content_hash, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (file_id, url, filename, scraped_at, content_hash, status)
    )
    conn.commit()

    logger.info(f"Inserted file record: {file_id} for URL: {url}")
    return file_id


def get_file_by_url(conn: sqlite3.Connection, url: str) -> Optional[sqlite3.Row]:
    """
    Retrieve a file record by its URL.

    Args:
        conn: Active database connection.
        url: The URL to look up.

    Returns:
        sqlite3.Row or None: The file record if found, None otherwise.
    """
    cursor = conn.execute(
        "SELECT * FROM files WHERE url = ?",
        (url,)
    )
    return cursor.fetchone()


def update_status(
    conn: sqlite3.Connection,
    file_id: str,
    status: str,
    error_message: Optional[str] = None
) -> bool:
    """
    Update the status of a file record.

    Args:
        conn: Active database connection.
        file_id: The file_id to update.
        status: The new status value.
        error_message: Optional error message (typically for "error" status).

    Returns:
        bool: True if a record was updated, False if no matching record found.
    """
    cursor = conn.execute(
        """
        UPDATE files
        SET status = ?, error_message = ?
        WHERE file_id = ?
        """,
        (status, error_message, file_id)
    )
    conn.commit()

    updated = cursor.rowcount > 0
    if updated:
        logger.info(f"Updated status of {file_id} to '{status}'")
    else:
        logger.warning(f"No file found with file_id: {file_id}")

    return updated


def get_files_by_status(
    conn: sqlite3.Connection,
    status: str
) -> list[sqlite3.Row]:
    """
    Retrieve all file records with a given status.

    Args:
        conn: Active database connection.
        status: The status to filter by.

    Returns:
        list[sqlite3.Row]: List of matching file records.
    """
    cursor = conn.execute(
        "SELECT * FROM files WHERE status = ?",
        (status,)
    )
    return cursor.fetchall()


def update_content_hash(
    conn: sqlite3.Connection,
    file_id: str,
    content_hash: str
) -> bool:
    """
    Update the content hash of a file record.

    Args:
        conn: Active database connection.
        file_id: The file_id to update.
        content_hash: The new content hash.

    Returns:
        bool: True if a record was updated, False if no matching record found.
    """
    scraped_at = datetime.utcnow().isoformat()
    cursor = conn.execute(
        """
        UPDATE files
        SET content_hash = ?, scraped_at = ?
        WHERE file_id = ?
        """,
        (content_hash, scraped_at, file_id)
    )
    conn.commit()

    updated = cursor.rowcount > 0
    if updated:
        logger.info(f"Updated content hash for {file_id}")

    return updated
