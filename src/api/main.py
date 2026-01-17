"""
FastAPI application for the Q for Mortals RAG API.

Provides endpoints for querying the Q4M documentation using semantic search.
"""

import logging
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
