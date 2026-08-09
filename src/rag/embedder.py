import time
from collections.abc import Sequence
from typing import Optional, Protocol

import numpy as np

from src.utils import get_logger


logger = get_logger(__name__)

QUERY_INSTRUCTION = (
    "Дан вопрос сотрудника компании, найди фрагменты внутренних регламентов, "
    "которые отвечают на этот вопрос"
)


class Embedder(Protocol):
    """The model is asymmetric: queries and documents are encoded differently."""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """(n, d) float32, L2-normalized."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """(d,) float32, L2-normalized."""
        ...


class GigaEmbedder:
    """
    ai-sage/Giga-Embeddings-instruct via transformers.

    The 3B model is loaded lazily on first use: fp16 + eager attention on CUDA, fp32 on CPU.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        hf_token: str = "",
        batch_size: int = 8,
        max_length: int = 4096,
    ) -> None:
        self._model_name = model_name
        self._device_setting = device
        self._hf_token = hf_token
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device: Optional[str] = None

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"])[0]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._device_setting == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self._device_setting
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        started = time.time()
        model = AutoModel.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation="eager",
            token=self._hf_token or None,
        )
        self._model = model.to(self._device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            token=self._hf_token or None,
        )
        logger.info(
            f"Embedding model {self._model_name} loaded on {self._device} "
            f"in {time.time() - started:.0f}s"
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        import torch

        parts: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch_size):
            batch = self._tokenizer(
                texts[start : start + self._batch_size],
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                embeddings = self._model(**batch, return_embeddings=True)
            parts.append(embeddings.cpu().to(torch.float32).numpy())

        vectors = np.vstack(parts)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        if vectors.dtype != np.float32:
            raise ValueError(f"FAISS requires float32 vectors, got {vectors.dtype}")
        return vectors
