"""
FastAPI application for the Q for Mortals RAG API.

Provides endpoints for querying the Q4M documentation using semantic search.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
    query: str = Field(..., description="The search query text", min_length=1)
    top_k: int = Field(
        default=5,
        description="Number of results to return",
        ge=1,
        le=20,
    )
    chapter: Optional[str] = Field(
        default=None,
        description="Optional chapter name to filter results",
    )


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


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Search for documents matching the query.

    Embeds the query text and searches the KDB.AI vector database
    for the most similar document chunks.

    Args:
        request: QueryRequest with query text and optional top_k.

    Returns:
        QueryResponse with ranked search results.
    """
    filter_info = f", chapter='{request.chapter}'" if request.chapter else ""
    logger.info(f"Query received: '{request.query[:50]}...' (top_k={request.top_k}{filter_info})")

    # Search for similar documents
    results = search_similar(
        query_text=request.query,
        top_k=request.top_k,
        chapter_filter=request.chapter,
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
