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
