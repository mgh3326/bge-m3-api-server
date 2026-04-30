# OpenAI-Compatible BGE-M3 Embedding Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local OpenAI-compatible embedding server wrapping BAAI/bge-m3 so Honcho can use `transport=openai` against `127.0.0.1:10631` instead of Gemini Embedding's quota-limited free tier.

**Architecture:** Single-process FastAPI application serving dense embeddings from FlagEmbedding's `BGEM3FlagModel`. The model loads at startup with a warmup call to prime CPU caches. Three endpoints (`/health`, `/v1/models`, `/v1/embeddings`) provide the minimum surface for OpenAI-compatible transport. Configuration via environment variables, loopback-only binding.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, FlagEmbedding (BGEM3FlagModel), PyTorch (CPU), Pydantic v2

---

## Known Risks

1. **Python version (MEDIUM):** `pyproject.toml` currently says `>=3.14`. PyTorch 2.11.0 does have 3.14 wheels and FlagEmbedding has no explicit Python constraint, but FlagEmbedding's transitive dependencies (compiled C extensions in tokenizers, safetensors, etc.) may not all ship 3.14 wheels for macOS x86_64 yet. **Resolution:** Task 1 downgrades to `>=3.12` and the venv is created with Python 3.12. If 3.14 works later, the constraint can be relaxed.

2. **Intel CPU first-token latency:** BGE-M3 is ~600MB. First inference after load can take several seconds on CPU. **Resolution:** Task 4 includes a warmup call during the `lifespan` startup event.

3. **Token counting approximation:** OpenAI's response requires `usage.prompt_tokens` and `usage.total_tokens`. BGE-M3's tokenizer is a SentencePiece/BPE tokenizer accessible via the underlying transformers model. **Resolution:** Task 4 uses the model's tokenizer for accurate counts, with a `len(text.split())` fallback if tokenizer access fails.

4. **Auth model:** `api_key=local` is a shared secret. Since the server binds to `127.0.0.1` only, this is acceptable for v0.1. **Resolution:** Task 4 implements constant-time comparison via `hmac.compare_digest`.

5. **Honcho DB recreation is destructive:** Switching from Gemini 1536-d to BGE-M3 1024-d requires dropping and recreating Honcho's vector store. **Resolution:** Task 8 README documents the procedure and warns not to drop the existing DB until the canary test passes.

---

## File Structure

```
bge-m3-api-server/
  pyproject.toml              -- Project metadata, dependencies, scripts entry point
  src/
    bge_m3_server/
      __init__.py             -- Package marker (empty)
      config.py               -- Pydantic Settings class, env var loading
      models.py               -- Pydantic request/response schemas
      embedding.py            -- BGEM3FlagModel wrapper (load, encode, tokenize)
      auth.py                 -- API key validation middleware
      routes.py               -- FastAPI route handlers
      app.py                  -- FastAPI app factory with lifespan
      __main__.py             -- uvicorn entry point
  tests/
    __init__.py               -- Package marker (empty)
    conftest.py               -- Shared fixtures (test client, mock model)
    test_config.py            -- Config loading tests
    test_models.py            -- Schema validation tests
    test_auth.py              -- Auth middleware tests
    test_embedding.py         -- Embedding wrapper unit tests
    test_routes.py            -- Route handler integration tests
  scripts/
    smoke_test.sh             -- curl-based smoke test
    benchmark.sh              -- Latency benchmark at batch sizes 1/4/8/16
  launchd/
    com.local.bge-m3-server.plist  -- macOS launchd template
  README.md                   -- Setup, usage, Honcho config, Intel notes
  .gitignore                  -- Python/venv/model cache ignores
  .env.example                -- Example environment variables
```

---

## Task 1: Project Scaffolding and Dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `src/bge_m3_server/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`
- Create: `.env.example`

- [ ] **Step 1: Update `pyproject.toml` with correct Python version and all dependencies**

```toml
[project]
name = "openai-compatible-bge-m3-server"
version = "0.1.0"
description = "Local OpenAI-compatible embedding server for BAAI/bge-m3"
requires-python = ">=3.12"
license = { text = "MIT" }
readme = "README.md"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "FlagEmbedding>=1.3",
    "torch>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
    "pytest-asyncio>=0.24",
]

[project.scripts]
bge-m3-server = "bge_m3_server.__main__:main"

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
*.egg

# Virtual environment
venv/
.venv/

# IDE
.idea/
.vscode/
*.swp

# Environment
.env

# Model cache
models/
*.bin
*.safetensors

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 3: Create `.env.example`**

```env
# Server binding
EMBEDDING_HOST=127.0.0.1
EMBEDDING_PORT=10631

# Model
EMBEDDING_MODEL=BAAI/bge-m3

# Auth (loopback-only, so a simple shared secret suffices)
EMBEDDING_API_KEY=local

# Batching
EMBEDDING_BATCH_SIZE=8

# Intel CPU threading (tune for your core count; 8 suits i9)
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
```

- [ ] **Step 4: Create empty `src/bge_m3_server/__init__.py`**

```python
```

- [ ] **Step 5: Create empty `tests/__init__.py`**

```python
```

- [ ] **Step 6: Create venv and install dependencies**

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Run: `python -c "import fastapi; import FlagEmbedding; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/bge_m3_server/__init__.py tests/__init__.py
git commit -m "chore: scaffold project with dependencies and structure"
```

---

## Task 2: Configuration Module

**Files:**
- Create: `src/bge_m3_server/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test for config loading**

```python
# tests/test_config.py
import os
import pytest


def test_default_config():
    """Config should load with sensible defaults."""
    from bge_m3_server.config import Settings

    settings = Settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 10631
    assert settings.model == "BAAI/bge-m3"
    assert settings.api_key == "local"
    assert settings.batch_size == 8


def test_config_from_env(monkeypatch):
    """Config should read from EMBEDDING_* env vars."""
    monkeypatch.setenv("EMBEDDING_HOST", "0.0.0.0")
    monkeypatch.setenv("EMBEDDING_PORT", "9999")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom/model")
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret123")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "16")

    from bge_m3_server.config import Settings

    settings = Settings()
    assert settings.host == "0.0.0.0"
    assert settings.port == 9999
    assert settings.model == "custom/model"
    assert settings.api_key == "secret123"
    assert settings.batch_size == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the config module**

```python
# src/bge_m3_server/config.py
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Server configuration loaded from environment variables."""

    host: str = "127.0.0.1"
    port: int = 10631
    model: str = "BAAI/bge-m3"
    api_key: str = "local"
    batch_size: int = 8

    model_config = {
        "env_prefix": "EMBEDDING_",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/bge_m3_server/config.py tests/test_config.py
git commit -m "feat: add configuration module with env var loading"
```

---

## Task 3: Pydantic Request/Response Schemas

**Files:**
- Create: `src/bge_m3_server/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests for schema validation**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the schema module**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/bge_m3_server/models.py tests/test_models.py
git commit -m "feat: add Pydantic request/response schemas for OpenAI compatibility"
```

---

## Task 4: Embedding Model Wrapper

**Files:**
- Create: `src/bge_m3_server/embedding.py`
- Create: `tests/test_embedding.py`

- [ ] **Step 1: Write the failing test for the embedding wrapper**

```python
# tests/test_embedding.py
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_embed_texts_returns_correct_shape():
    """embed_texts should return a list of 1024-d float lists."""
    from bge_m3_server.embedding import EmbeddingModel

    model = EmbeddingModel.__new__(EmbeddingModel)
    mock_bgem3 = MagicMock()
    mock_bgem3.encode.return_value = {
        "dense_vecs": np.random.rand(2, 1024).astype(np.float32)
    }
    model._model = mock_bgem3
    model._tokenizer = None

    result = model.embed_texts(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == 1024
    assert all(isinstance(x, float) for x in result[0])


def test_embed_texts_single_string():
    """embed_texts should handle a single string by wrapping it."""
    from bge_m3_server.embedding import EmbeddingModel

    model = EmbeddingModel.__new__(EmbeddingModel)
    mock_bgem3 = MagicMock()
    mock_bgem3.encode.return_value = {
        "dense_vecs": np.random.rand(1, 1024).astype(np.float32)
    }
    model._model = mock_bgem3
    model._tokenizer = None

    result = model.embed_texts(["single text"])
    assert len(result) == 1
    assert len(result[0]) == 1024


def test_count_tokens_with_tokenizer():
    """count_tokens should use the tokenizer when available."""
    from bge_m3_server.embedding import EmbeddingModel

    model = EmbeddingModel.__new__(EmbeddingModel)
    mock_tokenizer = MagicMock()
    # tokenizer.encode returns token IDs; len of that is the count
    mock_tokenizer.encode.side_effect = lambda t: list(range(len(t.split()) + 2))
    model._tokenizer = mock_tokenizer

    count = model.count_tokens(["hello world", "foo"])
    # "hello world" -> 4 tokens, "foo" -> 3 tokens = 7
    assert isinstance(count, int)
    assert count > 0


def test_count_tokens_fallback_without_tokenizer():
    """count_tokens should fall back to word splitting when no tokenizer."""
    from bge_m3_server.embedding import EmbeddingModel

    model = EmbeddingModel.__new__(EmbeddingModel)
    model._tokenizer = None

    count = model.count_tokens(["hello world", "foo bar baz"])
    assert count == 5  # 2 + 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedding.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the embedding wrapper**

```python
# src/bge_m3_server/embedding.py
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# BGE-M3 always produces 1024-d dense vectors
EMBEDDING_DIMENSION = 1024


class EmbeddingModel:
    """Wrapper around BGEM3FlagModel for dense embedding."""

    def __init__(self, model_id: str, use_fp16: bool = False) -> None:
        """Load the BGE-M3 model.

        Args:
            model_id: HuggingFace model identifier (e.g. "BAAI/bge-m3").
            use_fp16: Use half-precision. Set False for CPU (Intel).
        """
        from FlagEmbedding import BGEM3FlagModel

        logger.info("Loading model %s (fp16=%s)...", model_id, use_fp16)
        start = time.monotonic()
        self._model = BGEM3FlagModel(model_id, use_fp16=use_fp16)
        elapsed = time.monotonic() - start
        logger.info("Model loaded in %.1fs", elapsed)

        # Try to get the tokenizer for accurate token counting
        self._tokenizer: Optional[object] = None
        try:
            self._tokenizer = self._model.tokenizer
            logger.info("Tokenizer acquired for token counting")
        except AttributeError:
            logger.warning(
                "Could not access model tokenizer; "
                "falling back to word-count approximation for usage.prompt_tokens"
            )

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to dense embedding vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of 1024-d float lists, one per input text.
        """
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense: np.ndarray = output["dense_vecs"]
        return dense.tolist()

    def count_tokens(self, texts: list[str]) -> int:
        """Count tokens across all texts.

        Uses the model's tokenizer if available, otherwise falls back
        to whitespace splitting.
        """
        if self._tokenizer is not None:
            total = 0
            for text in texts:
                tokens = self._tokenizer.encode(text)
                total += len(tokens)
            return total
        # Fallback: whitespace-split word count
        return sum(len(t.split()) for t in texts)

    def warmup(self) -> float:
        """Run a dummy encode to warm CPU caches.

        Returns:
            Warmup time in milliseconds.
        """
        start = time.monotonic()
        self._model.encode(
            ["warmup"],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("Warmup completed in %.0fms", elapsed_ms)
        return elapsed_ms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embedding.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/bge_m3_server/embedding.py tests/test_embedding.py
git commit -m "feat: add BGE-M3 embedding model wrapper with tokenizer-based counting"
```

---

## Task 5: API Key Auth Middleware

**Files:**
- Create: `src/bge_m3_server/auth.py`
- Create: `tests/test_auth.py`

- [ ] **Step 1: Write the failing test for auth**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_auth.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the auth module**

```python
# src/bge_m3_server/auth.py
import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)


def create_api_key_dependency(expected_key: str):
    """Create a FastAPI dependency that validates Bearer token API keys.

    Uses hmac.compare_digest for constant-time comparison to prevent
    timing attacks (defense-in-depth even though we bind to loopback).
    """

    async def verify_api_key(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None, Depends(security)
        ] = None,
    ) -> bool:
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header",
            )
        if not hmac.compare_digest(credentials.credentials, expected_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        return True

    return Depends(verify_api_key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_auth.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/bge_m3_server/auth.py tests/test_auth.py
git commit -m "feat: add API key auth with constant-time comparison"
```

---

## Task 6: Route Handlers

**Files:**
- Create: `src/bge_m3_server/routes.py`
- Create: `tests/test_routes.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write conftest with shared fixtures**

```python
# tests/conftest.py
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_embedding_model():
    """Create a mock EmbeddingModel that returns deterministic results."""
    model = MagicMock()
    model.dimension = 1024

    def fake_embed(texts):
        return np.random.RandomState(42).rand(len(texts), 1024).tolist()

    def fake_count(texts):
        return sum(len(t.split()) for t in texts)

    model.embed_texts = MagicMock(side_effect=fake_embed)
    model.count_tokens = MagicMock(side_effect=fake_count)
    return model


@pytest.fixture
def test_client(mock_embedding_model):
    """Create a TestClient with the mock model injected."""
    from bge_m3_server.app import create_app
    from bge_m3_server.config import Settings

    settings = Settings()
    app = create_app(settings=settings, model=mock_embedding_model)
    return TestClient(app)
```

- [ ] **Step 2: Write the failing route tests**

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_routes.py -v`
Expected: FAIL (routes.py and app.py don't exist yet)

- [ ] **Step 4: Write the routes module**

```python
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
```

- [ ] **Step 5: Write the app factory**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_routes.py -v`
Expected: 7 passed

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (config: 2, models: 6, auth: 4, embedding: 4, routes: 7 = 23 total)

- [ ] **Step 8: Commit**

```bash
git add src/bge_m3_server/routes.py src/bge_m3_server/app.py tests/conftest.py tests/test_routes.py
git commit -m "feat: add route handlers and app factory with lifespan model loading"
```

---

## Task 7: Entry Point and Server Runner

**Files:**
- Create: `src/bge_m3_server/__main__.py`

- [ ] **Step 1: Write the entry point**

```python
# src/bge_m3_server/__main__.py
import logging
import sys


def main() -> None:
    """Run the BGE-M3 embedding server."""
    import uvicorn

    from .config import get_settings

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    settings = get_settings()

    uvicorn.run(
        "bge_m3_server.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the entry point is importable**

Run: `python -c "from bge_m3_server.__main__ import main; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/bge_m3_server/__main__.py
git commit -m "feat: add uvicorn entry point with logging configuration"
```

---

## Task 8: Scripts, launchd, and README

**Files:**
- Create: `scripts/smoke_test.sh`
- Create: `scripts/benchmark.sh`
- Create: `launchd/com.local.bge-m3-server.plist`
- Create: `README.md`

- [ ] **Step 1: Write the smoke test script**

```bash
#!/usr/bin/env bash
# scripts/smoke_test.sh — Smoke test for the BGE-M3 embedding server
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:10631}"
API_KEY="${2:-local}"

echo "=== Smoke Test: BGE-M3 Embedding Server ==="
echo "Target: $BASE_URL"
echo ""

# Test 1: Health check (no auth required)
echo "--- Test 1: GET /health ---"
HEALTH=$(curl -sf "$BASE_URL/health")
echo "$HEALTH" | python3 -m json.tool
DIM=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['embedding_dimension'])")
if [ "$DIM" != "1024" ]; then
    echo "FAIL: expected dimension 1024, got $DIM"
    exit 1
fi
echo "PASS: dimension is 1024"
echo ""

# Test 2: Models list
echo "--- Test 2: GET /v1/models ---"
curl -sf "$BASE_URL/v1/models" \
    -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
echo "PASS"
echo ""

# Test 3: Single string embedding
echo "--- Test 3: POST /v1/embeddings (single string) ---"
RESP=$(curl -sf "$BASE_URL/v1/embeddings" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, world!", "model": "BAAI/bge-m3"}')
EMB_DIM=$(echo "$RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data'][0]['embedding']))")
if [ "$EMB_DIM" != "1024" ]; then
    echo "FAIL: expected 1024-d embedding, got $EMB_DIM"
    exit 1
fi
echo "PASS: embedding is 1024-d"
echo ""

# Test 4: List input embedding
echo "--- Test 4: POST /v1/embeddings (list input) ---"
RESP=$(curl -sf "$BASE_URL/v1/embeddings" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input": ["first text", "second text"], "model": "BAAI/bge-m3"}')
COUNT=$(echo "$RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))")
if [ "$COUNT" != "2" ]; then
    echo "FAIL: expected 2 embeddings, got $COUNT"
    exit 1
fi
echo "PASS: got 2 embeddings"
echo ""

# Test 5: Auth rejection
echo "--- Test 5: Auth rejection (no key) ---"
HTTP_CODE=$(curl -so /dev/null -w "%{http_code}" "$BASE_URL/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d '{"input": "test", "model": "BAAI/bge-m3"}')
if [ "$HTTP_CODE" != "401" ]; then
    echo "FAIL: expected 401, got $HTTP_CODE"
    exit 1
fi
echo "PASS: got 401"
echo ""

echo "=== All smoke tests passed ==="
```

- [ ] **Step 2: Write the benchmark script**

```bash
#!/usr/bin/env bash
# scripts/benchmark.sh — Latency benchmark for the BGE-M3 embedding server
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:10631}"
API_KEY="${2:-local}"

echo "=== Benchmark: BGE-M3 Embedding Server ==="
echo "Target: $BASE_URL"
echo ""

# Generate N copies of a sample sentence
sample_text="The quick brown fox jumps over the lazy dog near the riverbank."

for BATCH_SIZE in 1 4 8 16; do
    # Build JSON array of $BATCH_SIZE copies
    INPUT_JSON=$(python3 -c "
import json
texts = ['$sample_text'] * $BATCH_SIZE
print(json.dumps({'input': texts, 'model': 'BAAI/bge-m3'}))
")

    echo "--- Batch size: $BATCH_SIZE ---"
    # Run 3 iterations, report each
    for i in 1 2 3; do
        TIME_MS=$(curl -sf -o /dev/null -w "%{time_total}" \
            "$BASE_URL/v1/embeddings" \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -d "$INPUT_JSON")
        TIME_MS=$(python3 -c "print(f'{float(\"$TIME_MS\") * 1000:.0f}')")
        echo "  Run $i: ${TIME_MS}ms"
    done
    echo ""
done

echo "=== Benchmark complete ==="
```

- [ ] **Step 3: Make scripts executable**

```bash
chmod +x scripts/smoke_test.sh scripts/benchmark.sh
```

- [ ] **Step 4: Write the launchd plist template**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.local.bge-m3-server</string>

    <key>ProgramArguments</key>
    <array>
        <!-- UPDATE THIS PATH to your venv's Python -->
        <string>/Users/robin/PycharmProjects/bge-m3-api-server/venv/bin/python</string>
        <string>-m</string>
        <string>bge_m3_server</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/robin/PycharmProjects/bge-m3-api-server</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>EMBEDDING_HOST</key>
        <string>127.0.0.1</string>
        <key>EMBEDDING_PORT</key>
        <string>10631</string>
        <key>EMBEDDING_MODEL</key>
        <string>BAAI/bge-m3</string>
        <key>EMBEDDING_API_KEY</key>
        <string>local</string>
        <key>OMP_NUM_THREADS</key>
        <string>8</string>
        <key>MKL_NUM_THREADS</key>
        <string>8</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/tmp/bge-m3-server.out.log</string>

    <key>StandardErrorPath</key>
    <string>/tmp/bge-m3-server.err.log</string>

    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
```

- [ ] **Step 5: Write README.md**

```markdown
# openai-compatible-bge-m3-server

Local OpenAI-compatible embedding server for [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3). Designed to replace quota-limited cloud embedding APIs (like Gemini Embedding's 1K RPD free tier) with a zero-quota local alternative.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/openai-compatible-bge-m3-server.git
cd openai-compatible-bge-m3-server
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run the server

```bash
# Set Intel CPU threading (adjust for your core count)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Start the server
python -m bge_m3_server
```

The server starts on `http://127.0.0.1:10631` by default. First startup downloads the model (~600MB) and takes 1-2 minutes.

### 3. Verify

```bash
# Health check
curl http://127.0.0.1:10631/health

# Generate embeddings
curl http://127.0.0.1:10631/v1/embeddings \
  -H "Authorization: Bearer local" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "model": "BAAI/bge-m3"}'
```

## Configuration

All settings are controlled via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_HOST` | `127.0.0.1` | Bind address (keep loopback for security) |
| `EMBEDDING_PORT` | `10631` | Bind port |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace model ID |
| `EMBEDDING_API_KEY` | `local` | API key for Bearer auth |
| `EMBEDDING_BATCH_SIZE` | `8` | Max batch size |
| `OMP_NUM_THREADS` | (system) | OpenMP threads (set to core count) |
| `MKL_NUM_THREADS` | (system) | MKL threads (set to core count) |

## Using with Honcho

Add these to your Honcho `.env`:

```env
EMBEDDING_MODEL_CONFIG__TRANSPORT=openai
EMBEDDING_MODEL_CONFIG__MODEL=BAAI/bge-m3
EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL=http://127.0.0.1:10631/v1
EMBEDDING_MODEL_CONFIG__OVERRIDES__API_KEY=local
EMBEDDING_VECTOR_DIMENSIONS=1024
VECTOR_STORE_DIMENSIONS=1024
```

### IMPORTANT: Database Recreation Required

BGE-M3 produces **1024-dimensional** vectors. If your Honcho instance previously used Gemini Embedding (1536-d) or another model with different dimensions, you **must** recreate the vector store:

1. **Back up your existing database** before proceeding
2. Stop Honcho
3. Drop and recreate the database (or the vector extension tables)
4. Restart Honcho with the new env vars above
5. Run a canary test: write a memory and read it back
6. Only after the canary passes, remove your old database backup

**Do NOT drop your existing database until you have confirmed the new configuration works end-to-end.**

## Using with the OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:10631/v1",
    api_key="local",
)

response = client.embeddings.create(
    input=["Your text here"],
    model="BAAI/bge-m3",
)

# Each embedding is 1024 dimensions
print(len(response.data[0].embedding))  # 1024
```

## Running as a macOS Service (launchd)

```bash
cp launchd/com.local.bge-m3-server.plist ~/Library/LaunchAgents/
# Edit the plist to verify paths match your installation
launchctl load ~/Library/LaunchAgents/com.local.bge-m3-server.plist
```

To stop:
```bash
launchctl unload ~/Library/LaunchAgents/com.local.bge-m3-server.plist
```

Logs: `/tmp/bge-m3-server.out.log` and `/tmp/bge-m3-server.err.log`

## Testing

```bash
# Unit tests
pytest tests/ -v

# Smoke test (server must be running)
./scripts/smoke_test.sh

# Benchmark
./scripts/benchmark.sh
```

## Intel Mac CPU Performance Notes

- BGE-M3 on Intel i9 CPU with `OMP_NUM_THREADS=8`: expect ~200-500ms per single embedding, ~1-2s for batch of 16
- First request after startup is slower (model warmup); the server runs a warmup call automatically
- `use_fp16=False` is required for CPU mode (fp16 requires GPU)
- Setting `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to your physical core count (not logical) typically gives best throughput

### Future Optimization Options

When CPU performance becomes a bottleneck, consider these options (not included in v0.1):

- **ONNX Runtime:** Convert the model to ONNX format for faster CPU inference. See `myeolinmalchi/bge-m3-fastapi` for a reference implementation.
- **OpenVINO:** Intel's inference engine, optimized for Intel CPUs. Can provide 2-4x speedup on Intel hardware.
- **Quantization:** INT8 quantization can reduce memory and improve throughput with minimal quality loss.

Benchmark with `./scripts/benchmark.sh` before and after any optimization to verify improvement.

## API Reference

### GET /health
No auth required. Returns model info.
```json
{"status": "ok", "model": "BAAI/bge-m3", "embedding_dimension": 1024}
```

### GET /v1/models
Requires Bearer auth. Returns OpenAI-compatible model list.

### POST /v1/embeddings
Requires Bearer auth. Accepts `input` as string or list of strings.
```json
{
  "input": "text to embed",
  "model": "BAAI/bge-m3"
}
```
Response:
```json
{
  "object": "list",
  "data": [{"object": "embedding", "embedding": [0.1, ...], "index": 0}],
  "model": "BAAI/bge-m3",
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

## License

MIT
```

- [ ] **Step 6: Commit**

```bash
git add scripts/ launchd/ README.md
git commit -m "docs: add smoke test, benchmark, launchd template, and README"
```

---

## Task 9: Integration Verification (Manual)

This task runs after the server is started for the first time. It cannot be automated in unit tests because it requires the real model.

**Prerequisites:** Server running at `http://127.0.0.1:10631` with the real BGE-M3 model loaded.

- [ ] **Step 1: Start the server**

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python -m bge_m3_server
```

Wait for the log line: `Model ready (warmup: XXXms)`

- [ ] **Step 2: Run smoke tests**

```bash
./scripts/smoke_test.sh
```

Expected: `All smoke tests passed`

- [ ] **Step 3: Run benchmark**

```bash
./scripts/benchmark.sh
```

Expected: Timing results for batch sizes 1, 4, 8, 16. Record these in the PR or commit message for baseline.

- [ ] **Step 4: Verify OpenAI Python client compatibility**

```bash
python3 -c "
from openai import OpenAI
client = OpenAI(base_url='http://127.0.0.1:10631/v1', api_key='local')
resp = client.embeddings.create(input=['test'], model='BAAI/bge-m3')
assert len(resp.data[0].embedding) == 1024
print(f'OK: {len(resp.data[0].embedding)}-d embedding')
print(f'Usage: {resp.usage}')
"
```

Expected: `OK: 1024-d embedding`

Note: This requires `pip install openai` in the test environment.

- [ ] **Step 5: Honcho canary test**

1. Configure Honcho with the env vars from the README
2. Start Honcho
3. Write a test memory via Honcho's API
4. Read back the memory and verify it includes the embedding-based retrieval
5. Only after this passes, proceed with full Honcho migration

**IMPORTANT:** Do not drop the existing Honcho database (with Gemini 1536-d embeddings) until this canary test passes. Keep the old DB as a rollback option.

- [ ] **Step 6: Final commit with any integration-test fixes**

```bash
git add -A
git commit -m "chore: integration test fixes (if any)"
```

---

## Success Criteria

- [ ] `curl http://127.0.0.1:10631/health` returns `{"status":"ok","model":"BAAI/bge-m3","embedding_dimension":1024}`
- [ ] `POST /v1/embeddings` accepts both string and list input
- [ ] Response works with OpenAI Python client (`base_url=http://127.0.0.1:10631/v1`, `api_key=local`)
- [ ] All embeddings are exactly 1024 dimensions
- [ ] `pytest tests/ -v` passes all unit/integration tests
- [ ] Honcho canary write/read memory test passes after DB rebuild
- [ ] README documents Intel CPU notes and future ONNX/OpenVINO optimization path
