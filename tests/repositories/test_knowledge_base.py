"""
The FAISS knowledge base over the real ``data/knowledge_base`` corpus.

The 3B embedding model is never loaded: the suite-wide ``fake_embedder`` is a
deterministic bag-of-stems stand-in that satisfies the same contract (float32,
L2-normalised), so ranking assertions below are about *retrieval wiring* — that
the right document wins for the right question — not about model quality.
"""

from pathlib import Path

import numpy as np
import pytest

from src.config import BASE_DIR
from src.exceptions import KnowledgeBaseUnavailableError
from src.rag.chunker import split_markdown
from src.repositories.knowledge_base import FaissKnowledgeBase


KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"


def expected_chunk_count(directory: Path) -> int:
    """Recompute the corpus size the same way ``load()`` does."""
    total = 0
    for path in sorted(directory.glob("*.md")):
        total += len(
            split_markdown(path.read_text(encoding="utf-8"), source=path.name, start_id=total)
        )
    return total


@pytest.fixture
def knowledge_base(fake_embedder) -> FaissKnowledgeBase:
    """Not loaded yet — tests that need an index await ``load()`` themselves."""
    return FaissKnowledgeBase(fake_embedder, KNOWLEDGE_BASE_DIR)


@pytest.fixture
async def loaded_knowledge_base(knowledge_base: FaissKnowledgeBase) -> FaissKnowledgeBase:
    await knowledge_base.load()
    return knowledge_base


# --- loading -------------------------------------------------------------


async def test_load_indexes_every_markdown_document_in_the_corpus(
    loaded_knowledge_base: FaissKnowledgeBase,
) -> None:
    expected = expected_chunk_count(KNOWLEDGE_BASE_DIR)

    count = await loaded_knowledge_base.count()

    assert expected > 0, "the corpus under data/knowledge_base is empty"
    assert count == expected


async def test_every_document_contributes_at_least_one_chunk(
    loaded_knowledge_base: FaissKnowledgeBase,
) -> None:
    found = await loaded_knowledge_base.search("оплата аккаунт сбой", k=await loaded_knowledge_base.count())

    sources = {chunk.chunk.source for chunk in found}
    assert sources == {path.name for path in KNOWLEDGE_BASE_DIR.glob("*.md")}


async def test_load_on_a_directory_without_documents_raises_file_not_found(
    fake_embedder, tmp_path: Path
) -> None:
    empty = FaissKnowledgeBase(fake_embedder, tmp_path)

    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        await empty.load()


# --- retrieval -----------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected_source"),
    [
        (
            "Как поменять банковскую карту для оплаты подписки?",
            "payment_methods.md",
        ),
        (
            "С меня дважды списали деньги за один заказ, хочу возврат",
            "payment_methods.md",
        ),
        (
            "Забыл пароль и потерял доступ к аккаунту, как восстановить?",
            "account_security.md",
        ),
        (
            "Я случайно прислал паспорт, что будет с моими персональными данными?",
            "account_security.md",
        ),
        (
            "Сайт отдаёт ошибку 502, у вас массовый сбой?",
            "incident_management.md",
        ),
    ],
    ids=[
        "change_card",
        "double_charge",
        "account_recovery",
        "pii_in_the_ticket",
        "outage",
    ],
)
async def test_the_right_document_is_retrieved_first(
    loaded_knowledge_base: FaissKnowledgeBase, question: str, expected_source: str
) -> None:
    found = await loaded_knowledge_base.search(question, k=4)

    assert found[0].chunk.source == expected_source


async def test_results_come_back_best_first(
    loaded_knowledge_base: FaissKnowledgeBase,
) -> None:
    found = await loaded_knowledge_base.search("оплата картой и подписка", k=4)

    scores = [chunk.score for chunk in found]
    assert scores == sorted(scores, reverse=True)


async def test_search_returns_at_most_k_chunks(
    loaded_knowledge_base: FaissKnowledgeBase,
) -> None:
    found = await loaded_knowledge_base.search("восстановление аккаунта", k=2)

    assert len(found) == 2


async def test_asking_for_more_chunks_than_exist_drops_the_padding(
    loaded_knowledge_base: FaissKnowledgeBase,
) -> None:
    total = await loaded_knowledge_base.count()

    found = await loaded_knowledge_base.search("восстановление аккаунта", k=total + 10)

    # FAISS pads a short result set with index -1; those rows must be filtered.
    assert len(found) == total


# --- the degradation contract -------------------------------------------


async def test_search_before_load_reports_the_knowledge_base_as_unavailable(
    knowledge_base: FaissKnowledgeBase,
) -> None:
    # A RuntimeError here would 500 the worker instead of routing the ticket
    # to an operator, so the exception type is part of the contract.
    with pytest.raises(KnowledgeBaseUnavailableError, match="not loaded") as caught:
        await knowledge_base.search("что угодно", k=4)

    assert caught.value.status_code == 503
    assert caught.value.code == "knowledge_base_unavailable"


async def test_ping_is_false_before_load_and_true_after(
    knowledge_base: FaissKnowledgeBase,
) -> None:
    assert await knowledge_base.ping() is False
    assert await knowledge_base.count() == 0

    await knowledge_base.load()

    assert await knowledge_base.ping() is True


# --- what actually goes into the index ----------------------------------


async def test_indexed_vectors_are_float32_and_l2_normalised(
    loaded_knowledge_base: FaissKnowledgeBase,
) -> None:
    index = loaded_knowledge_base._index
    assert index is not None

    vectors = index.reconstruct_n(0, index.ntotal)

    assert vectors.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(vectors, axis=1), np.ones(index.ntotal), rtol=0, atol=1e-6
    )


async def test_cosine_scores_stay_inside_the_unit_range(
    loaded_knowledge_base: FaissKnowledgeBase,
) -> None:
    found = await loaded_knowledge_base.search("оплата подписки картой", k=4)

    # A zero vector would normalise to NaN and FAISS would rank on nonsense.
    assert not any(np.isnan(chunk.score) for chunk in found)
    assert all(-1.0 <= chunk.score <= 1.0 for chunk in found)
