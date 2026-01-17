"""
FastAPI application for the Q for Mortals RAG API.

Provides endpoints for querying the Q4M documentation using semantic search.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.api.query import search_similar

logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("config/config.yaml")) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# Load configuration
config = load_config()
api_config = config.get("api", {})

# Create FastAPI app
app = FastAPI(
    title="Q for Mortals RAG API",
    description="Semantic search API for Q for Mortals documentation",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        Dict with status and version info.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "q4m-rag-api",
    }


# Request/Response models for /query endpoint
class QueryRequest(BaseModel):
    """Request model for the query endpoint."""
    query: str = Field(..., description="The search query text", min_length=1, max_length=1000)
    top_k: int = Field(
        default=5,
        description="Number of results to return",
        ge=1,
        le=20,
    )
    chapter: Optional[str] = Field(
        default=None,
        description="Optional chapter name to filter results",
        max_length=200,
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_whitespace(cls, v: str) -> str:
        """Validate that query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class SearchResult(BaseModel):
    """A single search result."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="The chunk text content")
    chapter: str = Field(..., description="Chapter name")
    heading: str = Field(..., description="Section heading")
    url: str = Field(..., description="Source URL")


class QueryResponse(BaseModel):
    """Response model for the query endpoint."""
    query: str = Field(..., description="The original query")
    results: list[SearchResult] = Field(..., description="Search results")
    count: int = Field(..., description="Number of results returned")
    chapter_filter: Optional[str] = Field(
        default=None,
        description="Chapter filter applied (if any)",
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(default=None, description="Additional error details")


# Query logging helper
query_logger = logging.getLogger("query_log")


def log_query(
    query: str,
    num_results: int,
    latency_ms: float,
    chapter_filter: Optional[str] = None,
    error: Optional[str] = None,
):
    """Log query information in JSON format.

    Args:
        query: The query text.
        num_results: Number of results returned.
        latency_ms: Query latency in milliseconds.
        chapter_filter: Chapter filter applied (if any).
        error: Error message (if query failed).
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query[:100],  # Truncate long queries
        "num_results": num_results,
        "latency_ms": round(latency_ms, 2),
        "chapter_filter": chapter_filter,
    }

    if error:
        log_entry["error"] = error
        log_entry["status"] = "error"
    else:
        log_entry["status"] = "success"

    # Log as JSON for easy parsing
    query_logger.info(json.dumps(log_entry))


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "KDB.AI connection error"},
    },
)
async def query_documents(request: QueryRequest):
    """Search for documents matching the query.

    Embeds the query text and searches the KDB.AI vector database
    for the most similar document chunks.

    Args:
        request: QueryRequest with query text and optional top_k.

    Returns:
        QueryResponse with ranked search results.

    Raises:
        HTTPException: On validation errors or KDB.AI connection issues.
    """
    filter_info = f", chapter='{request.chapter}'" if request.chapter else ""
    logger.info(f"Query received: '{request.query[:50]}...' (top_k={request.top_k}{filter_info})")

    # Start timing
    start_time = time.perf_counter()

    try:
        # Search for similar documents
        results = search_similar(
            query_text=request.query,
            top_k=request.top_k,
            chapter_filter=request.chapter,
        )
    except ConnectionError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        log_query(
            query=request.query,
            num_results=0,
            latency_ms=latency_ms,
            chapter_filter=request.chapter,
            error="connection_error",
        )
        logger.error(f"KDB.AI connection error: {e}")
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error="connection_error",
                message="Unable to connect to KDB.AI database",
                details=str(e),
            ).model_dump(),
        )
    except ValueError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        log_query(
            query=request.query,
            num_results=0,
            latency_ms=latency_ms,
            chapter_filter=request.chapter,
            error="invalid_query",
        )
        logger.error(f"Value error during query: {e}")
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="invalid_query",
                message=str(e),
            ).model_dump(),
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        log_query(
            query=request.query,
            num_results=0,
            latency_ms=latency_ms,
            chapter_filter=request.chapter,
            error="internal_error",
        )
        logger.error(f"Unexpected error during query: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="internal_error",
                message="An unexpected error occurred",
                details=str(e),
            ).model_dump(),
        )

    # Format results
    formatted_results = []
    for result in results:
        formatted_results.append(SearchResult(
            chunk_id=result.get("chunk_id", ""),
            text=result.get("text", ""),
            chapter=result.get("chapter", ""),
            heading=result.get("heading", ""),
            url=result.get("url", ""),
        ))

    # Calculate latency and log success
    latency_ms = (time.perf_counter() - start_time) * 1000
    log_query(
        query=request.query,
        num_results=len(formatted_results),
        latency_ms=latency_ms,
        chapter_filter=request.chapter,
    )
    logger.info(f"Query completed: {len(formatted_results)} results in {latency_ms:.2f}ms")

    return QueryResponse(
        query=request.query,
        results=formatted_results,
        count=len(formatted_results),
        chapter_filter=request.chapter,
    )


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("Q4M RAG API starting up...")
    logger.info(f"API host: {api_config.get('host', '0.0.0.0')}")
    logger.info(f"API port: {api_config.get('port', 8000)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("Q4M RAG API shutting down...")


def main():
    """Run the API server."""
    import uvicorn

    # Configure logging
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_format = log_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.basicConfig(level=log_level, format=log_format)

    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)

    logger.info(f"Starting Q4M RAG API on {host}:{port}")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
