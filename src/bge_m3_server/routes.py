# src/bge_m3_server/routes.py
import logging
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

logger = logging.getLogger(__name__)

# These will be set by create_app via configure_routes
_model: EmbeddingModel
_settings: Settings

# Per-line preview length so the log fits one terminal row but still tells you
# at a glance which text was embedded (typo / ranking / semantic / etc.).
_PREVIEW_LEN = 80


def _preview(text: str, length: int = _PREVIEW_LEN) -> str:
    """Single-line preview of an embedding input. Newlines collapsed so each
    log line stays one row in `tail -f`."""
    one_line = text.replace("\n", " ⏎ ").replace("\t", " ")
    if len(one_line) <= length:
        return one_line
    return one_line[:length] + "…"


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
        t_start = time.monotonic()

        # Normalize input to list
        texts = (
            [request.input] if isinstance(request.input, str) else request.input
        )

        n = len(texts)
        total_chars = sum(len(t) for t in texts)
        avg_chars = total_chars / n if n else 0
        max_chars = max((len(t) for t in texts), default=0)
        first_preview = _preview(texts[0]) if texts else ""

        # Encode (timed separately so we can see model vs overhead)
        t_infer_start = time.monotonic()
        embeddings = model.embed_texts(texts)
        infer_ms = (time.monotonic() - t_infer_start) * 1000

        # Count tokens
        token_count = model.count_tokens(texts)

        total_ms = (time.monotonic() - t_start) * 1000

        logger.info(
            "[embed] n=%d chars=%d (avg=%.0f max=%d) tokens=%d infer=%.0fms total=%.0fms first=%r",
            n, total_chars, avg_chars, max_chars, token_count,
            infer_ms, total_ms, first_preview,
        )

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
