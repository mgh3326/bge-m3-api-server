# src/bge_m3_server/routes.py
import time

from fastapi import APIRouter, Depends

from .auth import create_api_key_dependency
from .config import Settings
from .embedding import EmbeddingModel
from .models import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    Usage,
)

# These will be set by create_app via configure_routes
_model: EmbeddingModel
_settings: Settings


def configure_routes(model: EmbeddingModel, settings: Settings) -> APIRouter:
    """Create and configure the API router with the given model and settings."""
    router = APIRouter()
    verify_key = create_api_key_dependency(settings.api_key)

    @router.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            model=settings.model,
            embedding_dimension=model.dimension,
            device=getattr(model, "device", "cpu"),
            fp16=getattr(model, "use_fp16", False),
        )

    @router.get("/v1/models", response_model=ModelListResponse)
    async def list_models(authorized: bool = verify_key):
        return ModelListResponse(
            object="list",
            data=[
                ModelInfo(
                    id=settings.model,
                    object="model",
                    created=0,
                    owned_by="BAAI",
                )
            ],
        )

    @router.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def create_embeddings(
        request: EmbeddingRequest,
        authorized: bool = verify_key,
    ):
        # Normalize input to list
        texts = (
            [request.input] if isinstance(request.input, str) else request.input
        )

        # Encode
        embeddings = model.embed_texts(texts)

        # Count tokens
        token_count = model.count_tokens(texts)

        return EmbeddingResponse(
            object="list",
            data=[
                EmbeddingData(
                    object="embedding",
                    embedding=emb,
                    index=i,
                )
                for i, emb in enumerate(embeddings)
            ],
            model=settings.model,
            usage=Usage(
                prompt_tokens=token_count,
                total_tokens=token_count,
            ),
        )

    return router
