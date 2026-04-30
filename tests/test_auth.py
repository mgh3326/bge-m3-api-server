# tests/test_auth.py
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


def _make_app_with_auth(api_key: str) -> FastAPI:
    """Create a minimal FastAPI app with auth middleware for testing."""
    from bge_m3_server.auth import create_api_key_dependency

    app = FastAPI()
    verify_key = create_api_key_dependency(api_key)

    @app.get("/protected")
    async def protected(authorized: bool = verify_key):
        return {"ok": True}

    return app


def test_valid_api_key():
    """Request with correct Bearer token should succeed."""
    app = _make_app_with_auth("local")
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": "Bearer local"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_missing_auth_header():
    """Request without Authorization header should return 401."""
    app = _make_app_with_auth("local")
    client = TestClient(app)
    resp = client.get("/protected")
    assert resp.status_code == 401


def test_wrong_api_key():
    """Request with wrong key should return 401."""
    app = _make_app_with_auth("local")
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_malformed_auth_header():
    """Request with non-Bearer auth should return 401."""
    app = _make_app_with_auth("local")
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert resp.status_code == 401
