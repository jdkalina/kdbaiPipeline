"""
KDB.AI client wrapper for vector database operations.

This module provides a clean interface to KDB.AI for storing and searching
document embeddings.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import kdbai_client as kdbai
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from config/config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class KDBAIClient:
    """
    Wrapper for KDB.AI client operations.

    Provides methods to connect, create tables, insert data, and search.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        Initialize the KDB.AI client.

        Args:
            endpoint: KDB.AI endpoint URL. Defaults to config/env variable.
            database: Database name. Defaults to config value.
        """
        config = load_config()
        kdbai_config = config.get("kdbai", {})

        # Endpoint from: arg > env var > config
        self.endpoint = (
            endpoint
            or os.environ.get("KDBAI_ENDPOINT")
            or kdbai_config.get("endpoint", "http://192.168.1.68:8082")
        )
        self.database_name = database or kdbai_config.get("database", "default")
        self.table_name = kdbai_config.get("table_name", "q4m_chunks")

        # Index configuration
        self.index_config = kdbai_config.get("index", {})
        self.index_name = self.index_config.get("name", "flat_index")
        self.index_type = self.index_config.get("type", "flat")
        self.index_metric = self.index_config.get("metric", "CS")

        # Embedding dimensions from embedding config
        embedding_config = config.get("embedding", {})
        self.dimensions = embedding_config.get("dimensions", 384)

        # Batch settings
        self.insert_batch_size = kdbai_config.get("insert_batch_size", 100)

        # Connection state
        self._session: Optional[kdbai.Session] = None
        self._db: Optional[Any] = None
        self._table: Optional[Any] = None

        logger.info(f"KDBAIClient initialized with endpoint={self.endpoint}, database={self.database_name}")

    def connect(self) -> "KDBAIClient":
        """
        Connect to KDB.AI server.

        Returns:
            Self for method chaining.

        Raises:
            ConnectionError: If connection fails.
        """
        try:
            logger.info(f"Connecting to KDB.AI at {self.endpoint}")
            self._session = kdbai.Session(endpoint=self.endpoint)
            self._db = self._session.database(self.database_name)
            logger.info(f"Connected to database '{self.database_name}'")
            return self
        except Exception as e:
            logger.error(f"Failed to connect to KDB.AI: {e}")
            raise ConnectionError(f"Failed to connect to KDB.AI: {e}") from e

    @property
    def session(self) -> kdbai.Session:
        """Get the active session, connecting if needed."""
        if self._session is None:
            self.connect()
        return self._session

    @property
    def db(self) -> Any:
        """Get the database handle, connecting if needed."""
        if self._db is None:
            self.connect()
        return self._db

    def create_table(
        self,
        table_name: Optional[str] = None,
        drop_if_exists: bool = False,
    ) -> Any:
        """
        Create a table for storing embeddings.

        Args:
            table_name: Table name. Defaults to config value.
            drop_if_exists: If True, drop existing table first.

        Returns:
            The created table handle.
        """
        table_name = table_name or self.table_name

        # Check if table exists
        existing_tables = [t.name for t in self.db.tables]

        if table_name in existing_tables:
            if drop_if_exists:
                logger.info(f"Dropping existing table '{table_name}'")
                self.db.table(table_name).drop()
            else:
                logger.info(f"Table '{table_name}' already exists, using existing")
                self._table = self.db.table(table_name)
                return self._table

        # Define schema
        schema = [
            {"name": "chunk_id", "type": "str"},
            {"name": "text", "type": "str"},
            {"name": "chapter", "type": "str"},
            {"name": "heading", "type": "str"},
            {"name": "url", "type": "str"},
            {"name": "file_id", "type": "str"},
            {"name": "embeddings", "type": "float32s"},
        ]

        # Define indexes
        indexes = [
            {
                "name": self.index_name,
                "type": self.index_type,
                "column": "embeddings",
                "params": {
                    "dims": self.dimensions,
                    "metric": self.index_metric,
                },
            }
        ]

        logger.info(f"Creating table '{table_name}' with {len(schema)} columns")
        self._table = self.db.create_table(table_name, schema=schema, indexes=indexes)
        logger.info(f"Table '{table_name}' created successfully")

        return self._table

    def get_table(self, table_name: Optional[str] = None) -> Any:
        """
        Get a reference to an existing table.

        Args:
            table_name: Table name. Defaults to config value.

        Returns:
            The table handle.

        Raises:
            ValueError: If table doesn't exist.
        """
        table_name = table_name or self.table_name

        existing_tables = [t.name for t in self.db.tables]
        if table_name not in existing_tables:
            raise ValueError(f"Table '{table_name}' does not exist")

        self._table = self.db.table(table_name)
        return self._table

    def insert_batch(self, data: list[dict]) -> int:
        """
        Insert a batch of records into the table.

        Args:
            data: List of dicts with keys matching schema columns.

        Returns:
            Number of records inserted.
        """
        if not data:
            logger.warning("Empty data provided to insert_batch")
            return 0

        if self._table is None:
            self._table = self.get_table()

        df = pd.DataFrame(data)
        self._table.insert(df)
        logger.debug(f"Inserted {len(data)} records")

        return len(data)

    def search(
        self,
        query_vector: list[float],
        n: int = 5,
        filter_expr: Optional[list] = None,
    ) -> list[dict]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector.
            n: Number of results to return.
            filter_expr: Optional filter expression, e.g., [("=", "chapter", "Tables")]

        Returns:
            List of matching records with similarity scores.
        """
        if self._table is None:
            self._table = self.get_table()

        search_kwargs = {
            "vectors": {self.index_name: [query_vector]},
            "n": n,
        }

        if filter_expr:
            search_kwargs["filter"] = filter_expr

        logger.debug(f"Searching for {n} similar vectors")
        results = self._table.search(**search_kwargs)

        # Convert results to list of dicts
        if isinstance(results, pd.DataFrame):
            records = results.to_dict("records")
        else:
            records = list(results)

        logger.debug(f"Found {len(records)} results")
        return records

    def close(self) -> None:
        """Close the connection to KDB.AI."""
        self._table = None
        self._db = None
        self._session = None
        logger.info("KDB.AI connection closed")


# Module-level singleton for convenience
_client: Optional[KDBAIClient] = None


def get_client() -> KDBAIClient:
    """
    Get a singleton KDB.AI client instance.

    Returns:
        Configured and connected KDBAIClient.
    """
    global _client
    if _client is None:
        _client = KDBAIClient()
        _client.connect()
    return _client


def connect(
    endpoint: Optional[str] = None,
    database: Optional[str] = None,
) -> KDBAIClient:
    """
    Create and connect a new KDB.AI client.

    Args:
        endpoint: KDB.AI endpoint URL.
        database: Database name.

    Returns:
        Connected KDBAIClient instance.
    """
    client = KDBAIClient(endpoint=endpoint, database=database)
    return client.connect()


def create_table(
    table_name: Optional[str] = None,
    drop_if_exists: bool = False,
) -> Any:
    """
    Create a table using the singleton client.

    Args:
        table_name: Table name.
        drop_if_exists: If True, drop existing table first.

    Returns:
        The created table handle.
    """
    return get_client().create_table(table_name=table_name, drop_if_exists=drop_if_exists)


def create_embeddings_table(drop_if_exists: bool = False) -> Any:
    """
    Create the embeddings table with the standard schema for Q4M chunks.

    This is a convenience function that creates a table with the following schema:
        - chunk_id (str): Unique identifier for each chunk
        - text (str): The actual text content of the chunk
        - chapter (str): Chapter name/title from the source document
        - heading (str): Section heading hierarchy (e.g., "Chapter > Section > Subsection")
        - url (str): Source URL where the content was scraped from
        - file_id (str): Reference to the file in the SQLite status database
        - embeddings (float32s): Vector embeddings for similarity search

    The table includes a flat index on the embeddings column using cosine similarity.

    Args:
        drop_if_exists: If True, drop existing table and recreate. Default False.

    Returns:
        The created table handle.

    Example:
        >>> from src.kdbai.client import create_embeddings_table
        >>> table = create_embeddings_table()
        >>> print(f"Table created: {table.name}")
    """
    return get_client().create_table(drop_if_exists=drop_if_exists)


def insert_batch(data: list[dict]) -> int:
    """
    Insert records using the singleton client.

    Args:
        data: List of records to insert.

    Returns:
        Number of records inserted.
    """
    return get_client().insert_batch(data)


def search(
    query_vector: list[float],
    n: int = 5,
    filter_expr: Optional[list] = None,
) -> list[dict]:
    """
    Search using the singleton client.

    Args:
        query_vector: Query embedding vector.
        n: Number of results.
        filter_expr: Optional filter.

    Returns:
        List of matching records.
    """
    return get_client().search(query_vector=query_vector, n=n, filter_expr=filter_expr)


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.DEBUG)

    print("Testing KDB.AI client...")

    # Test connection
    client = KDBAIClient()
    client.connect()
    print(f"Connected to {client.endpoint}")

    # List existing tables
    tables = [t.name for t in client.db.tables]
    print(f"Existing tables: {tables}")

    print("KDB.AI client test completed!")
