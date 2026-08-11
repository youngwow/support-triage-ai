"""
The draft graph, built for real and run end to end.

``build_draft_graph`` is called with the production node implementations; only
the two collaborators outside the process — the vector index and the LLM — are
scripted. What is asserted is the final state, because that is the whole
contract the draft service consumes.
"""

from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent.graph import build_draft_graph
from src.agent.nodes import (
    _LOW_CONFIDENCE_REASON,
    _NO_CONTEXT_REASON,
    _UNGROUNDED_REASON,
)
from src.agent.prompts import GENERATOR_SYSTEM_PROMPT
from src.agent.schemas import DraftState, GroundedDraft, initial_draft_state
from src.config import Settings
from src.models.domain import Category, DocumentChunk, RetrievedChunk, RiskLevel
from src.repositories.knowledge_base import AbstractKnowledgeBase


MIN_RETRIEVAL_SCORE = 0.5
MIN_DRAFT_CONFIDENCE = 0.6

PAYMENTS_TEXT = "Чтобы изменить привязанную карту, зайдите в «Способы оплаты»."
REFUND_TEXT = "Жалобы на двойное списание получают приоритет High."


def chunk(
    *, score: float, source: str = "payment_methods.md", text: str = PAYMENTS_TEXT, id: int = 0
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(id=id, source=source, heading="Смена карты", text=text),
        score=score,
    )


def draft(*, is_grounded: bool = True, confidence: float = 0.9) -> GroundedDraft:
    return GroundedDraft(
        answer="Замените карту в разделе «Способы оплаты».",
        sources=["payment_methods.md"],
        is_grounded=is_grounded,
        confidence=confidence,
    )


class PresetKnowledgeBase(AbstractKnowledgeBase):
    """Returns a fixed ranked list and records what it was asked for."""

    def __init__(self, *chunks: RetrievedChunk) -> None:
        self._chunks = list(chunks)
        self.queries: list[tuple[str, int]] = []

    async def search(self, query: str, *, k: int) -> list[RetrievedChunk]:
        self.queries.append((query, k))
        return self._chunks[:k]

    async def count(self) -> int:
        return len(self._chunks)

    async def ping(self) -> bool:
        return True


class ScriptedLLM:
    """Hands back queued answers and keeps every prompt it was given.

    ``queue`` staying full is the evidence that no generation call was made.
    """

    def __init__(self, *answers: Any) -> None:
        self.queue: deque[Any] = deque(answers)
        self.calls: list[SimpleNamespace] = []

    async def generate_structured(self, *, system: str, user: str, schema: type) -> Any:
        self.calls.append(SimpleNamespace(system=system, user=user, schema=schema))
        answer = self.queue.popleft()
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def settings() -> Settings:
    return Settings(
        min_retrieval_score=MIN_RETRIEVAL_SCORE,
        min_draft_confidence=MIN_DRAFT_CONFIDENCE,
        retrieval_top_k=4,
    )


async def run(
    *,
    knowledge_base: AbstractKnowledgeBase,
    llm: ScriptedLLM,
    settings: Settings,
    question: str = "Как поменять привязанную карту?",
    category: Category = "billing/faq",
    risk: RiskLevel = "low",
) -> DraftState:
    graph = build_draft_graph(knowledge_base=knowledge_base, llm=llm, settings=settings)
    return await graph.ainvoke(
        initial_draft_state(
            ticket_id="11111111-1111-1111-1111-111111111111",
            question=question,
            category=category,
            risk=risk,
        )
    )


# --- the happy path ------------------------------------------------------


async def test_grounded_confident_draft_reaches_the_end_without_escalation(
    settings: Settings,
) -> None:
    llm = ScriptedLLM(draft())

    state = await run(
        knowledge_base=PresetKnowledgeBase(chunk(score=0.9)), llm=llm, settings=settings
    )

    assert state["escalated"] is False
    assert state["escalation_reason"] is None
    assert state["answer"] == "Замените карту в разделе «Способы оплаты»."
    assert state["sources"] == ["payment_methods.md"]
    assert len(llm.calls) == 1


async def test_the_retrieved_question_and_top_k_come_from_the_settings(
    settings: Settings,
) -> None:
    knowledge_base = PresetKnowledgeBase(chunk(score=0.9))

    await run(
        knowledge_base=knowledge_base,
        llm=ScriptedLLM(draft()),
        settings=settings,
        question="Где посмотреть статус заказа?",
    )

    assert knowledge_base.queries == [("Где посмотреть статус заказа?", 4)]


# --- the retrieval gate --------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expect_escalated"),
    [
        (MIN_RETRIEVAL_SCORE - 0.01, True),
        (MIN_RETRIEVAL_SCORE, False),
        (MIN_RETRIEVAL_SCORE + 0.01, False),
    ],
    ids=["below_the_floor", "exactly_at_the_floor", "above_the_floor"],
)
async def test_the_retrieval_floor_decides_whether_generation_runs(
    settings: Settings, score: float, expect_escalated: bool
) -> None:
    llm = ScriptedLLM(draft())

    state = await run(
        knowledge_base=PresetKnowledgeBase(chunk(score=score)), llm=llm, settings=settings
    )

    assert state["escalated"] is expect_escalated
    assert len(llm.calls) == (0 if expect_escalated else 1)


@pytest.mark.parametrize(
    "found",
    [[], [chunk(score=0.2)]],
    ids=["index_returned_nothing", "everything_below_the_floor"],
)
async def test_nothing_relevant_escalates_without_spending_a_generation_call(
    settings: Settings, found: list[RetrievedChunk]
) -> None:
    unused = draft()
    llm = ScriptedLLM(unused)

    state = await run(knowledge_base=PresetKnowledgeBase(*found), llm=llm, settings=settings)

    assert state["escalated"] is True
    assert state["escalation_reason"] == _NO_CONTEXT_REASON
    assert state["answer"] is None
    assert state["generation"] is None
    # The queue is untouched, so no generation call was ever made.
    assert list(llm.queue) == [unused]
    assert llm.calls == []


async def test_only_chunks_above_the_floor_survive_into_the_state(
    settings: Settings,
) -> None:
    keeper = chunk(score=0.8, id=1)
    dropped = chunk(score=0.1, id=2, source="incident_management.md", text=REFUND_TEXT)

    state = await run(
        knowledge_base=PresetKnowledgeBase(keeper, dropped),
        llm=ScriptedLLM(draft()),
        settings=settings,
    )

    assert state["chunks"] == [keeper]


# --- the generation gate -------------------------------------------------


@pytest.mark.parametrize(
    ("generation", "expected_reason"),
    [
        (draft(is_grounded=False, confidence=0.99), _UNGROUNDED_REASON),
        (
            draft(confidence=MIN_DRAFT_CONFIDENCE - 0.01),
            _LOW_CONFIDENCE_REASON.format(confidence=MIN_DRAFT_CONFIDENCE - 0.01),
        ),
        (draft(confidence=0.0), _LOW_CONFIDENCE_REASON.format(confidence=0.0)),
    ],
    ids=["ungrounded", "just_below_the_confidence_floor", "no_confidence_at_all"],
)
async def test_a_draft_the_model_does_not_stand_behind_is_escalated(
    settings: Settings, generation: GroundedDraft, expected_reason: str
) -> None:
    state = await run(
        knowledge_base=PresetKnowledgeBase(chunk(score=0.9)),
        llm=ScriptedLLM(generation),
        settings=settings,
    )

    assert state["escalated"] is True
    assert state["escalation_reason"] == expected_reason
    # The draft is kept: an operator still sees what the model proposed.
    assert state["answer"] == generation.answer


async def test_confidence_exactly_at_the_floor_is_accepted(settings: Settings) -> None:
    state = await run(
        knowledge_base=PresetKnowledgeBase(chunk(score=0.9)),
        llm=ScriptedLLM(draft(confidence=MIN_DRAFT_CONFIDENCE)),
        settings=settings,
    )

    assert state["escalated"] is False


# --- the prompt ----------------------------------------------------------


async def test_the_prompt_carries_the_fragments_and_the_ticket_text(
    settings: Settings,
) -> None:
    keeper = chunk(score=0.9, id=1)
    dropped = chunk(score=0.1, id=2, source="incident_management.md", text=REFUND_TEXT)
    llm = ScriptedLLM(draft())

    await run(
        knowledge_base=PresetKnowledgeBase(keeper, dropped),
        llm=llm,
        settings=settings,
        question="Как поменять карту?",
        category="billing/faq",
        risk="medium",
    )

    call = llm.calls[0]
    assert call.system == GENERATOR_SYSTEM_PROMPT
    assert call.schema is GroundedDraft
    assert "[payment_methods.md]" in call.user
    assert PAYMENTS_TEXT in call.user
    assert "Как поменять карту?" in call.user
    assert "billing/faq" in call.user
    assert "medium" in call.user
    # A fragment that failed the floor must not reach the generator.
    assert REFUND_TEXT not in call.user
