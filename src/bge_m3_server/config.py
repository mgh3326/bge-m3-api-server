# src/bge_m3_server/config.py
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Server configuration loaded from environment variables.

    All variables are prefixed with EMBEDDING_ (e.g. EMBEDDING_DEVICE).
    """

    # Network
    host: str = "127.0.0.1"
    port: int = 10631

    # Model
    model: str = "BAAI/bge-m3"

    # Auth
    api_key: str = "local"

    # Inference
    # device: "auto" (default — pick cuda > mps > cpu), "cpu", "cuda", "mps"
    device: str = "auto"
    # use_fp16: None = auto (True on cuda/mps, False on cpu); set explicitly to override
    use_fp16: Optional[bool] = None
    # internal batch size used by FlagEmbedding.encode
    encode_batch_size: int = 12
    # max token length per input (BGE-M3 supports up to 8192; lower = faster)
    max_length: int = 8192

    # Legacy alias kept for backward compatibility with prior versions
    batch_size: int = 8

    model_config = {
        "env_prefix": "EMBEDDING_",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
