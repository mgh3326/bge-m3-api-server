# tests/test_routes.py
import pytest


def test_health_endpoint(test_client):
    """GET /health should return model info and dimension."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == "BAAI/bge-m3"
    assert body["embedding_dimension"] == 1024


def test_models_endpoint(test_client):
    """GET /v1/models should return an OpenAI-style model list."""
    resp = test_client.get(
        "/v1/models",
        headers={"Authorization": "Bearer local"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == "BAAI/bge-m3"
    assert body["data"][0]["object"] == "model"


def test_embeddings_single_string(test_client):
    """POST /v1/embeddings with a single string should return one embedding."""
    resp = test_client.post(
        "/v1/embeddings",
        json={"input": "hello world", "model": "BAAI/bge-m3"},
        headers={"Authorization": "Bearer local"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) == 1024
    assert body["model"] == "BAAI/bge-m3"
    assert "prompt_tokens" in body["usage"]
    assert "total_tokens" in body["usage"]


def test_embeddings_list_input(test_client):
    """POST /v1/embeddings with a list should return multiple embeddings."""
    resp = test_client.post(
        "/v1/embeddings",
        json={"input": ["hello", "world", "foo"], "model": "BAAI/bge-m3"},
        headers={"Authorization": "Bearer local"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 3
    for i, item in enumerate(body["data"]):
        assert item["index"] == i
        assert len(item["embedding"]) == 1024


def test_embeddings_requires_auth(test_client):
    """POST /v1/embeddings without auth should return 401."""
    resp = test_client.post(
        "/v1/embeddings",
        json={"input": "hello", "model": "BAAI/bge-m3"},
    )
    assert resp.status_code == 401


def test_embeddings_empty_input_rejected(test_client):
    """POST /v1/embeddings with empty list should return 422."""
    resp = test_client.post(
        "/v1/embeddings",
        json={"input": [], "model": "BAAI/bge-m3"},
        headers={"Authorization": "Bearer local"},
    )
    assert resp.status_code == 422


def test_health_no_auth_required(test_client):
    """GET /health should not require authentication."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
