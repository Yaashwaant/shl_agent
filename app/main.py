"""
FastAPI application factory and lifespan events.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.api.routes import router

settings = get_settings()

# Setup logging before anything else
setup_logging(log_level=settings.log_level, log_file="logs/app.log")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    Startup: Load catalog, build vector index.
    Shutdown: Cleanup resources.
    """
    logger.info("=" * 60)
    logger.info("SHL Assessment Agent — Starting up")
    logger.info("=" * 60)

    # Lazy import to avoid circular imports
    from app.services.vector_store import get_vector_store

    vector_store = get_vector_store()

    catalog_path = Path(settings.catalog_json_path)
    if catalog_path.exists():
        logger.info(f"Loading catalog from {catalog_path}")
        vector_store.load_catalog()
        vector_store.build_index()  # No-op if already indexed
        logger.info("Pre-loading cross-encoder model for reranking...")
        vector_store._get_cross_encoder()  # Download model at startup, not on first request
    else:
        logger.warning(
            f"Catalog JSON not found at {catalog_path}. "
            "Run 'python scripts/scrape_catalog.py' to populate the catalog. "
            "Service will start but recommendations will be empty."
        )

    # Pre-compile the agent graph
    from app.services.agent import get_agent
    get_agent()

    logger.info("Startup complete — service is ready")
    yield

    logger.info("Shutting down SHL Assessment Agent")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SHL Assessment Advisor API",
        description=(
            "Conversational agent that helps hiring managers select the right "
            "SHL assessments through multi-turn dialogue. "
            "Built with FastAPI + LangGraph + ChromaDB."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow all origins for evaluation harness
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info(f"→ {request.method} {request.url.path}")
        response = await call_next(request)
        logger.info(f"← {response.status_code} {request.url.path}")
        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Please try again."},
        )

    # Include routers
    app.include_router(router, prefix="")

    return app


app = create_app()
