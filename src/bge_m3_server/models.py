# src/bge_m3_server/models.py
from typing import Union

from pydantic import BaseModel, field_validator


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    input: Union[str, list[str]]
    model: str
    encoding_format: str = "float"

    @field_validator("input")
    @classmethod
    def input_not_empty(cls, v: Union[str, list[str]]) -> Union[str, list[str]]:
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("input list must not be empty")
        if isinstance(v, str) and len(v.strip()) == 0:
            raise ValueError("input string must not be empty")
        return v


class EmbeddingData(BaseModel):
    """Single embedding in the response."""

    object: str = "embedding"
    embedding: list[float]
    index: int


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage


class ModelInfo(BaseModel):
    """Single model entry for /v1/models."""

    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response."""

    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model: str
    embedding_dimension: int
