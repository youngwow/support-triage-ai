"""Tests for FaissKnowledgeBase over the real data/ corpus with a fake embedder."""

import pytest

from src.config import BASE_DIR
from src.models.domain import RetrievedChunk
from src.rag.chunker import split_markdown
from src.repositories.knowledge_base import FaissKnowledgeBase


DATA_DIR = BASE_DIR / "data"

LEAVE_QUESTION = "За сколько дней нужно предупреждать об отпуске?"


@pytest.fixture
def knowledge_base(fake_embedder) -> FaissKnowledgeBase:
    return FaissKnowledgeBase(fake_embedder, DATA_DIR)


@pytest.fixture
async def loaded_knowledge_base(knowledge_base) -> FaissKnowledgeBase:
    await knowledge_base.load()
    return knowledge_base


async def test_load_indexes_every_chunk_of_every_document(loaded_knowledge_base):
    expected = sum(
        len(split_markdown(path.read_text(encoding="utf-8"), source=path.name))
        for path in DATA_DIR.glob("*.md")
    )

    assert expected > 0
    assert await loaded_knowledge_base.count() == expected


async def test_ping_flips_from_false_to_true_on_load(knowledge_base):
    assert await knowledge_base.ping() is False

    await knowledge_base.load()

    assert await knowledge_base.ping() is True


async def test_search_before_load_raises_runtime_error(knowledge_base):
    with pytest.raises(RuntimeError, match="not loaded"):
        await knowledge_base.search(LEAVE_QUESTION, k=3)


async def test_load_on_directory_without_markdown_raises(fake_embedder, tmp_path):
    knowledge_base = FaissKnowledgeBase(fake_embedder, tmp_path)

    with pytest.raises(FileNotFoundError, match="No markdown documents"):
        await knowledge_base.load()

    assert await knowledge_base.ping() is False
    assert await knowledge_base.count() == 0


async def test_leave_question_retrieves_leave_policy_first(loaded_knowledge_base):
    results = await loaded_knowledge_base.search(LEAVE_QUESTION, k=3)

    assert len(results) == 3
    assert all(isinstance(result, RetrievedChunk) for result in results)
    assert results[0].chunk.source == "leave_policy.md"
    assert isinstance(results[0].score, float)


async def test_scores_are_descending_best_first(loaded_knowledge_base):
    results = await loaded_knowledge_base.search(LEAVE_QUESTION, k=5)

    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


async def test_k_larger_than_corpus_returns_every_chunk_once(loaded_knowledge_base):
    count = await loaded_knowledge_base.count()

    results = await loaded_knowledge_base.search(LEAVE_QUESTION, k=count + 1000)

    # FAISS pads missing neighbours with index -1; those must be filtered out,
    # and the ids prove start_id chaining across documents left no duplicates.
    assert len(results) == count
    assert {result.chunk.id for result in results} == set(range(count))
