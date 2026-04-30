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
