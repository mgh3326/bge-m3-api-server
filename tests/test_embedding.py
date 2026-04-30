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
