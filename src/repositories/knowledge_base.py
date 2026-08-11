import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from src.exceptions import KnowledgeBaseUnavailableError
from src.models.domain import DocumentChunk, RetrievedChunk
from src.rag.chunker import split_markdown
from src.rag.embedder import Embedder
from src.utils import get_logger


logger = get_logger(__name__)


class AbstractKnowledgeBase(ABC):
    """Vector search over the support knowledge base."""

    @abstractmethod
    async def search(self, query: str, *, k: int) -> list[RetrievedChunk]:
        """Top-``k`` chunks by similarity, best first."""

    @abstractmethod
    async def count(self) -> int:
        """Number of indexed chunks."""

    @abstractmethod
    async def ping(self) -> bool:
        """True when the index is built and non-empty."""

    async def load(self) -> None:
        """Optional warm-up hook awaited once from the app lifespan."""
        return None


class FaissKnowledgeBase(AbstractKnowledgeBase):
    """
    Chunks ``data_dir/*.md``, embeds them and serves cosine search via FAISS.

    ``IndexFlatIP`` over L2-normalized vectors is exact cosine similarity. At
    the PoC's corpus size an exact index is simply correct; the target design
    swaps it for an ANN index once the corpus stops fitting in memory.

    The index is built at startup and never persisted, so every boot re-embeds
    the corpus. Acceptable for three documents, not for fifty thousand.
    """

    def __init__(self, embedder: Embedder, data_dir: Path) -> None:
        self._embedder = embedder
        self._data_dir = data_dir
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: list[DocumentChunk] = []

    async def load(self) -> None:
        paths = sorted(self._data_dir.glob("*.md"))
        if not paths:
            raise FileNotFoundError(f"No markdown documents found in {self._data_dir}")

        chunks: list[DocumentChunk] = []
        for path in paths:
            chunks.extend(
                split_markdown(
                    path.read_text(encoding="utf-8"),
                    source=path.name,
                    start_id=len(chunks),
                )
            )

        vectors: np.ndarray = await asyncio.to_thread(
            self._embedder.embed_documents, [chunk.text for chunk in chunks]
        )
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self._chunks = chunks
        self._index = index
        logger.info(
            f"Knowledge base ready: {len(chunks)} chunks from {len(paths)} documents"
        )

    async def search(self, query: str, *, k: int) -> list[RetrievedChunk]:
        if self._index is None:
            # An AppError rather than a RuntimeError: the draft path catches it
            # and routes the ticket to an operator instead of 500-ing.
            raise KnowledgeBaseUnavailableError("Knowledge base is not loaded")
        query_vector = await asyncio.to_thread(self._embedder.embed_query, query)
        scores, indices = self._index.search(query_vector.reshape(1, -1), k)
        return [
            RetrievedChunk(chunk=self._chunks[index], score=float(score))
            for score, index in zip(scores[0], indices[0])
            if index != -1
        ]

    async def count(self) -> int:
        return len(self._chunks)

    async def ping(self) -> bool:
        return self._index is not None and self._index.ntotal > 0
