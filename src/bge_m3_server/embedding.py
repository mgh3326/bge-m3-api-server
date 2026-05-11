# src/bge_m3_server/embedding.py
import logging
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# BGE-M3 always produces 1024-d dense vectors
EMBEDDING_DIMENSION = 1024


def select_device(preferred: Optional[str] = None) -> str:
    """Pick a torch device.

    "auto" / None  : prefer cuda, then mps, fall back to cpu.
    "mps"|"cuda"|"cpu" : honor the explicit choice (still validate availability).
    """
    import torch

    pref = (preferred or "auto").lower()
    if pref == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("device=cuda requested but CUDA unavailable; falling back to auto")
        pref = "auto"
    if pref == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        logger.warning("device=mps requested but MPS unavailable; falling back to auto")
        pref = "auto"
    if pref == "cpu":
        return "cpu"

    # auto
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


class EmbeddingModel:
    """Wrapper around BGEM3FlagModel for dense embedding."""

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None,
        encode_batch_size: int = 12,
        max_length: int = 8192,
    ) -> None:
        """Load the BGE-M3 model.

        Args:
            model_id: HuggingFace model identifier (e.g. "BAAI/bge-m3").
            device: "auto" (default), "mps", "cuda", or "cpu".
            use_fp16: half-precision. None = auto (True on cuda/mps, False on cpu).
            encode_batch_size: internal batch size used by FlagEmbedding.encode.
            max_length: max token length per input. Default 8192 matches
                BGE-M3's training window; lower values (e.g. 1024) trade
                support for very long inputs for ~5-8x faster inference on
                MPS due to attention's O(L^2) cost.
        """
        from FlagEmbedding import BGEM3FlagModel

        self._device = select_device(device)
        if use_fp16 is None:
            use_fp16 = self._device in ("cuda", "mps")
        self._use_fp16 = use_fp16
        self._encode_batch_size = encode_batch_size
        self._max_length = max_length

        # Some MPS ops still fall back to CPU; setting this avoids hard errors.
        if self._device == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        logger.info(
            "Loading model %s (device=%s, fp16=%s, batch=%d, max_length=%d)...",
            model_id, self._device, use_fp16, encode_batch_size, max_length,
        )
        start = time.monotonic()
        self._model = BGEM3FlagModel(
            model_id,
            use_fp16=use_fp16,
            devices=[self._device],
            batch_size=encode_batch_size,
            passage_max_length=max_length,
            return_sparse=False,
            return_colbert_vecs=False,
        )
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

    @property
    def device(self) -> str:
        return self._device

    @property
    def use_fp16(self) -> bool:
        return self._use_fp16

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to dense embedding vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of 1024-d float lists, one per input text.
        """
        output = self._model.encode(
            texts,
            batch_size=self._encode_batch_size,
            max_length=self._max_length,
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
        """Run a dummy encode to warm caches / JIT compile MPS kernels.

        Returns:
            Warmup time in milliseconds.
        """
        start = time.monotonic()
        self._model.encode(
            ["warmup"],
            batch_size=1,
            max_length=64,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("Warmup completed in %.0fms", elapsed_ms)
        return elapsed_ms
