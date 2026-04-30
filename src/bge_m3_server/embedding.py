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
