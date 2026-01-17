"""
Embedding pipeline orchestrator.

This module orchestrates the full embedding pipeline:
1. Poll SQLite database for files with status='scraped'
2. Read the scraped markdown content from data/raw/
3. Chunk the content using heading-aware chunking
4. Generate embeddings for each chunk
5. Insert chunks with embeddings into KDB.AI
6. Update file status to 'embedded'
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

import yaml

from src.db import init_db, get_files_by_status, update_status
from src.embedding.chunker import chunk_by_headings, Chunk
from src.embedding.embedder import get_embedder
from src.kdbai.client import get_client, create_embeddings_table
from src.scraper.storage import read_content

logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("config/config.yaml")) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_chapter_from_heading(chunk: Chunk) -> str:
    """Extract chapter name from chunk heading hierarchy.

    Args:
        chunk: A Chunk object with heading metadata.

    Returns:
        The H1 heading (chapter name) or a default value.
    """
    if chunk.heading_h1:
        return chunk.heading_h1
    return "Unknown Chapter"


def prepare_chunk_record(
    chunk: Chunk,
    embedding: list[float],
    file_id: str,
    url: str,
) -> dict:
    """Prepare a chunk record for insertion into KDB.AI.

    Args:
        chunk: The Chunk object with text and metadata.
        embedding: The embedding vector for the chunk.
        file_id: Reference to the file in SQLite database.
        url: Source URL of the content.

    Returns:
        Dict matching the KDB.AI table schema.
    """
    return {
        "chunk_id": str(uuid.uuid4()),
        "text": chunk.text,
        "chapter": extract_chapter_from_heading(chunk),
        "heading": chunk.heading_hierarchy or chunk.heading or "",
        "url": url,
        "file_id": file_id,
        "embeddings": embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
    }


def process_file(
    file_id: str,
    url: str,
    filename: str,
    raw_dir: Path,
    embedder,
    kdbai_client,
) -> tuple[int, int]:
    """Process a single scraped file through the embedding pipeline.

    Args:
        file_id: Unique identifier for the file.
        url: Source URL of the content.
        filename: Name of the file in raw_dir.
        raw_dir: Directory containing scraped markdown files.
        embedder: FastEmbed embedder instance.
        kdbai_client: KDB.AI client instance.

    Returns:
        Tuple of (chunks_created, chunks_inserted).

    Raises:
        FileNotFoundError: If the file cannot be found.
        Exception: For embedding or insertion errors.
    """
    # Read the scraped content
    content = read_content(url, raw_dir)
    if content is None:
        # Try reading by filename directly
        file_path = raw_dir / filename
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(f"Cannot find file for URL: {url}")

    logger.info(f"  Reading file: {filename} ({len(content)} chars)")

    # Chunk the content
    logger.debug(f"  Chunking content...")
    chunks = chunk_by_headings(content, source_file=filename)
    if not chunks:
        logger.warning(f"  No chunks created for {filename}")
        return 0, 0

    logger.info(f"  Chunking complete: {len(chunks)} chunks created")

    # Generate embeddings for all chunks
    chunk_texts = [chunk.text for chunk in chunks]
    logger.debug(f"  Generating embeddings for {len(chunk_texts)} chunks...")
    embeddings = embedder.generate_embeddings(chunk_texts)

    logger.info(f"  Embedding complete: {len(embeddings)} embeddings generated")

    # Prepare records for KDB.AI
    records = [
        prepare_chunk_record(chunk, embedding, file_id, url)
        for chunk, embedding in zip(chunks, embeddings)
    ]

    # Insert into KDB.AI
    logger.debug(f"  Inserting {len(records)} records into KDB.AI...")
    inserted = kdbai_client.insert_chunks(records)
    logger.info(f"  Insert complete: {inserted} chunks inserted to KDB.AI")

    return len(chunks), inserted


def run_pipeline(
    config_path: Path = Path("config/config.yaml"),
    create_table: bool = True,
    drop_existing_table: bool = False,
) -> dict:
    """Run the full embedding pipeline.

    Polls the database for files with status='scraped', processes each one
    through chunking and embedding, inserts into KDB.AI, and updates status.

    Args:
        config_path: Path to configuration file.
        create_table: If True, create the KDB.AI table if it doesn't exist.
        drop_existing_table: If True, drop and recreate the table (for fresh start).

    Returns:
        Dict with statistics: files_processed, files_failed, total_chunks, total_inserted.
    """
    config = load_config(config_path)

    # Setup paths
    db_path = Path(config.get("database", {}).get("path", "data/status.db"))
    raw_dir = Path("data/raw")

    # Log pipeline configuration
    embedding_config = config.get("embedding", {})
    kdbai_config = config.get("kdbai", {})
    logger.info("=" * 60)
    logger.info("Starting Embedding Pipeline")
    logger.info(f"  Model: {embedding_config.get('model_name', 'BAAI/bge-small-en-v1.5')}")
    logger.info(f"  Chunk size: {embedding_config.get('chunk_size', 512)} tokens")
    logger.info(f"  Chunk overlap: {embedding_config.get('chunk_overlap', 50)} tokens")
    logger.info(f"  KDB.AI table: {kdbai_config.get('table_name', 'q4m_chunks')}")
    logger.info("=" * 60)

    # Initialize database connection
    conn = init_db(db_path)

    # Get files ready for embedding
    files = get_files_by_status(conn, "scraped")

    if not files:
        logger.info("No files with status='scraped' found. Nothing to process.")
        return {
            "files_processed": 0,
            "files_failed": 0,
            "total_chunks": 0,
            "total_inserted": 0,
        }

    logger.info(f"Found {len(files)} files to process")

    # Initialize embedder (lazy loads model on first use)
    embedder = get_embedder(config_path=config_path)

    # Initialize KDB.AI client and create table if needed
    kdbai_client = get_client()

    if create_table:
        kdbai_client.create_table(drop_if_exists=drop_existing_table)

    # Process each file
    stats = {
        "files_processed": 0,
        "files_failed": 0,
        "total_chunks": 0,
        "total_inserted": 0,
    }

    total_files = len(files)
    for file_index, file_row in enumerate(files, 1):
        file_id = file_row["file_id"]
        url = file_row["url"]
        filename = file_row["filename"]

        logger.info(f"[{file_index}/{total_files}] Processing: {url}")

        try:
            chunks_created, chunks_inserted = process_file(
                file_id=file_id,
                url=url,
                filename=filename,
                raw_dir=raw_dir,
                embedder=embedder,
                kdbai_client=kdbai_client,
            )

            # Update status to 'embedded'
            update_status(conn, file_id, "embedded")

            stats["files_processed"] += 1
            stats["total_chunks"] += chunks_created
            stats["total_inserted"] += chunks_inserted

            logger.info(
                f"[{file_index}/{total_files}] Completed: {filename} "
                f"({chunks_created} chunks, {chunks_inserted} inserted)"
            )

        except Exception as e:
            logger.error(f"[{file_index}/{total_files}] Failed to process {url}: {e}")
            update_status(conn, file_id, "error", error_message=str(e))
            stats["files_failed"] += 1

    # Summary
    logger.info("=" * 60)
    logger.info("Embedding Pipeline Complete")
    logger.info(f"  Files processed: {stats['files_processed']}")
    logger.info(f"  Files failed: {stats['files_failed']}")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  Total inserted: {stats['total_inserted']}")
    logger.info("=" * 60)

    return stats


def main():
    """Main entry point for the embedding pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the embedding pipeline to process scraped content."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--drop-table",
        action="store_true",
        help="Drop and recreate the KDB.AI table (fresh start)",
    )
    parser.add_argument(
        "--no-create-table",
        action="store_true",
        help="Don't create table if it doesn't exist",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    config = load_config(args.config)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = config.get("logging", {}).get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.basicConfig(level=log_level, format=log_format)

    # Run the pipeline
    stats = run_pipeline(
        config_path=args.config,
        create_table=not args.no_create_table,
        drop_existing_table=args.drop_table,
    )

    # Exit with error code if any failures
    if stats["files_failed"] > 0:
        exit(1)


if __name__ == "__main__":
    main()
