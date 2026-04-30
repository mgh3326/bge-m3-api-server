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
