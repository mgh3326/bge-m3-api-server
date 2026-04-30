# src/bge_m3_server/app.py
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from .config import Settings, get_settings
from .embedding import EmbeddingModel
from .routes import configure_routes

logger = logging.getLogger(__name__)


def create_app(
    settings: Optional[Settings] = None,
    model: Optional[EmbeddingModel] = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        settings: Optional settings override (for testing).
        model: Optional model override (for testing with mocks).
    """
    if settings is None:
        settings = get_settings()

    # If model is provided (testing), skip lifespan model loading
    if model is not None:
        app = FastAPI(
            title="BGE-M3 Embedding Server",
            description="OpenAI-compatible embedding API for BAAI/bge-m3",
            version="0.1.0",
        )
        router = configure_routes(model=model, settings=settings)
        app.include_router(router)
        return app

    # Production: load model in lifespan
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting BGE-M3 Embedding Server")
        logger.info("Loading model: %s", settings.model)

        embedding_model = EmbeddingModel(
            model_id=settings.model,
            use_fp16=False,  # CPU mode on Intel Mac
        )

        # Warmup to prime CPU caches
        warmup_ms = embedding_model.warmup()
        logger.info("Model ready (warmup: %.0fms)", warmup_ms)

        router = configure_routes(model=embedding_model, settings=settings)
        app.include_router(router)

        yield

        logger.info("Shutting down BGE-M3 Embedding Server")

    app = FastAPI(
        title="BGE-M3 Embedding Server",
        description="OpenAI-compatible embedding API for BAAI/bge-m3",
        version="0.1.0",
        lifespan=lifespan,
    )

    return app
