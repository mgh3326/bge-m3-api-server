# tests/test_models.py
import pytest


def test_embedding_request_string_input():
    """Request should accept a single string as input."""
    from bge_m3_server.models import EmbeddingRequest

    req = EmbeddingRequest(input="hello world", model="BAAI/bge-m3")
    assert req.input == "hello world"
    assert req.model == "BAAI/bge-m3"


def test_embedding_request_list_input():
    """Request should accept a list of strings as input."""
    from bge_m3_server.models import EmbeddingRequest

    req = EmbeddingRequest(input=["hello", "world"], model="BAAI/bge-m3")
    assert req.input == ["hello", "world"]


def test_embedding_request_rejects_empty_list():
    """Request should reject an empty list."""
    from bge_m3_server.models import EmbeddingRequest

    with pytest.raises(Exception):
        EmbeddingRequest(input=[], model="BAAI/bge-m3")


def test_embedding_response_shape():
    """Response should match OpenAI's embedding response shape."""
    from bge_m3_server.models import EmbeddingData, EmbeddingResponse, Usage

    data = EmbeddingData(
        object="embedding",
        embedding=[0.1] * 1024,
        index=0,
    )
    usage = Usage(prompt_tokens=5, total_tokens=5)
    resp = EmbeddingResponse(
        object="list",
        data=[data],
        model="BAAI/bge-m3",
        usage=usage,
    )
    d = resp.model_dump()
    assert d["object"] == "list"
    assert len(d["data"]) == 1
    assert d["data"][0]["object"] == "embedding"
    assert len(d["data"][0]["embedding"]) == 1024
    assert d["data"][0]["index"] == 0
    assert d["usage"]["prompt_tokens"] == 5
    assert d["usage"]["total_tokens"] == 5


def test_model_info_shape():
    """ModelInfo should serialize for /v1/models response."""
    from bge_m3_server.models import ModelInfo, ModelListResponse

    info = ModelInfo(
        id="BAAI/bge-m3",
        object="model",
        created=1700000000,
        owned_by="BAAI",
    )
    resp = ModelListResponse(object="list", data=[info])
    d = resp.model_dump()
    assert d["data"][0]["id"] == "BAAI/bge-m3"


def test_health_response_shape():
    """HealthResponse should include model and dimension."""
    from bge_m3_server.models import HealthResponse

    h = HealthResponse(
        status="ok",
        model="BAAI/bge-m3",
        embedding_dimension=1024,
    )
    assert h.embedding_dimension == 1024
