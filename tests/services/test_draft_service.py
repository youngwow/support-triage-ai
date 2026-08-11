"""
The asynchronous draft path as a whole.

The service is built with the *real* graph over a scripted LLM and a preset
knowledge base, so these tests exercise retrieval, generation, the escalation
routes and the degradation fallback together — the failure modes only appear
when the pieces are wired up.

Two invariants carry most of the weight:

* nothing that happens to the provider or the index escapes as an exception;
* ``draft_ready`` requires *both* a clean run and ``auto_send_allowed``.
"""

from collections import deque
from typing import Any, Optional
from uuid import UUID, uuid4

import pytest

from src.agent.graph import build_draft_graph
from src.agent.nodes import _NO_CONTEXT_REASON
from src.agent.prompts import (
    DEGRADED_WITH_CONTEXT,
    DEGRADED_WITHOUT_CONTEXT,
    ESCALATION_NOTE,
)
from src.agent.schemas import GroundedDraft
from src.config import Settings
from src.exceptions import KnowledgeBaseUnavailableError, LLMUnavailableError
from src.models.domain import (
    Category,
    DocumentChunk,
    RetrievedChunk,
    RiskLevel,
    Ticket,
    TriageDecision,
)
from src.repositories.audit_log import AbstractAuditLog, InMemoryAuditLog
from src.repositories.draft_queue import DraftQueue
from src.repositories.knowledge_base import AbstractKnowledgeBase
from src.repositories.ticket_repository import (
    AbstractTicketRepository,
    InMemoryTicketRepository,
)
from src.services.draft_service import DraftService


MIN_RETRIEVAL_SCORE = 0.5
MIN_DRAFT_CONFIDENCE = 0.6

TICKET_TEXT = "Не могу поменять банковскую карту для оплаты подписки."
FRAGMENT_TEXT = "Чтобы изменить привязанную карту, зайдите в «Способы оплаты»."
SOURCE = "payment_methods.md"

ANSWER = "Замените карту в разделе «Способы оплаты»."


def retrieved(score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(id=0, source=SOURCE, heading="Смена карты", text=FRAGMENT_TEXT),
        score=score,
    )


def grounded_draft(*, is_grounded: bool = True, confidence: float = 0.9) -> GroundedDraft:
    return GroundedDraft(
        answer=ANSWER, sources=[SOURCE], is_grounded=is_grounded, confidence=confidence
    )


class PresetKnowledgeBase(AbstractKnowledgeBase):
    """A fixed ranked list, or a hard failure when the index is down."""

    def __init__(
        self, *chunks: RetrievedChunk, fails_with: Optional[Exception] = None
    ) -> None:
        self._chunks = list(chunks)
        self._fails_with = fails_with
        self.queries: list[str] = []

    async def search(self, query: str, *, k: int) -> list[RetrievedChunk]:
        self.queries.append(query)
        if self._fails_with is not None:
            raise self._fails_with
        return self._chunks[:k]

    async def count(self) -> int:
        return len(self._chunks)

    async def ping(self) -> bool:
        return self._fails_with is None


class ScriptedLLM:
    """Replays queued answers; an exception in the queue is raised instead."""

    def __init__(self, *answers: Any) -> None:
        self.queue: deque[Any] = deque(answers)
        self.calls: int = 0

    async def generate_structured(self, *, system: str, user: str, schema: type) -> Any:
        self.calls += 1
        answer = self.queue.popleft()
        if isinstance(answer, BaseException):
            raise answer
        return answer


class CountingQueue(DraftQueue):
    """A real queue that records how often the consumer acknowledged an item."""

    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize)
        self.acknowledged = 0

    def task_done(self) -> None:
        super().task_done()
        self.acknowledged += 1


def make_ticket(
    *,
    auto_send_allowed: bool = True,
    triaged: bool = True,
    category: Category = "billing/faq",
    risk: RiskLevel = "low",
    text: str = TICKET_TEXT,
) -> Ticket:
    decision = (
        TriageDecision(
            category=category,
            category_confidence=0.92,
            risk=risk,
            route="auto_answer" if auto_send_allowed else "operator_queue",
            auto_send_allowed=auto_send_allowed,
            classifier="llm",
        )
        if triaged
        else None
    )
    return Ticket(
        id=uuid4(), channel="chat", text=text, status="queued", decision=decision
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        min_retrieval_score=MIN_RETRIEVAL_SCORE,
        min_draft_confidence=MIN_DRAFT_CONFIDENCE,
        retrieval_top_k=4,
    )


@pytest.fixture
def tickets() -> AbstractTicketRepository:
    return InMemoryTicketRepository()


@pytest.fixture
def audit() -> AbstractAuditLog:
    return InMemoryAuditLog()


@pytest.fixture
def queue() -> CountingQueue:
    return CountingQueue(maxsize=10)


def build_service(
    *,
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: DraftQueue,
    settings: Settings,
    knowledge_base: AbstractKnowledgeBase,
    llm: ScriptedLLM,
) -> DraftService:
    return DraftService(
        tickets=tickets,
        audit=audit,
        knowledge_base=knowledge_base,
        queue=queue,
        graph=build_draft_graph(
            knowledge_base=knowledge_base, llm=llm, settings=settings
        ),
        settings=settings,
    )


async def only_audit_record(audit: AbstractAuditLog, ticket_id: UUID) -> dict[str, Any]:
    records = await audit.for_ticket(ticket_id)
    assert [record.event for record in records] == ["drafted"]
    return records[0].details


# --- a clean run ---------------------------------------------------------


async def test_a_grounded_draft_on_an_auto_send_ticket_becomes_draft_ready(
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: CountingQueue,
    settings: Settings,
) -> None:
    ticket = await tickets.add(make_ticket(auto_send_allowed=True))
    service = build_service(
        tickets=tickets,
        audit=audit,
        queue=queue,
        settings=settings,
        knowledge_base=PresetKnowledgeBase(retrieved()),
        llm=ScriptedLLM(grounded_draft()),
    )

    result = await service.process(ticket.id)

    assert result is not None
    assert result.status == "draft_ready"
    assert result.draft is not None
    assert result.draft.text == ANSWER
    assert result.draft.sources == (SOURCE,)
    assert result.draft.is_grounded is True
    assert result.draft.confidence == pytest.approx(0.9)
    assert result.draft.degraded is False


async def test_the_drafted_ticket_is_saved_and_audited(
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: CountingQueue,
    settings: Settings,
) -> None:
    ticket = await tickets.add(make_ticket(auto_send_allowed=True))
    service = build_service(
        tickets=tickets,
        audit=audit,
        queue=queue,
        settings=settings,
        knowledge_base=PresetKnowledgeBase(retrieved()),
        llm=ScriptedLLM(grounded_draft()),
    )

    await service.process(ticket.id)

    stored = await tickets.get(ticket.id)
    assert stored is not None
    assert stored.status == "draft_ready"
    assert stored.draft is not None and stored.draft.text == ANSWER

    details = await only_audit_record(audit, ticket.id)
    assert details == {
        "status": "draft_ready",
        "degraded": False,
        "escalated": False,
        "is_grounded": True,
        "confidence": pytest.approx(0.9),
        "sources": [SOURCE],
        "reason": "",
    }


async def test_the_same_draft_waits_for_an_operator_when_auto_send_is_not_allowed(
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: CountingQueue,
    settings: Settings,
) -> None:
    ticket = await tickets.add(make_ticket(auto_send_allowed=False))
    service = build_service(
        tickets=tickets,
        audit=audit,
        queue=queue,
        settings=settings,
        knowledge_base=PresetKnowledgeBase(retrieved()),
        llm=ScriptedLLM(grounded_draft()),
    )

    result = await service.process(ticket.id)

    assert result is not None
    assert result.status == "awaiting_operator"
    # The human still gets the suggestion; only the send decision changes.
    assert result.draft is not None
    assert result.draft.text == ANSWER
    assert result.draft.is_grounded is True
    assert result.draft.degraded is False


@pytest.mark.parametrize(
    "generation",
    [
        grounded_draft(is_grounded=False),
        grounded_draft(confidence=MIN_DRAFT_CONFIDENCE - 0.01),
    ],
    ids=["ungrounded_draft", "unconfident_draft"],
)
async def test_an_escalated_run_never_reaches_draft_ready(
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: CountingQueue,
    settings: Settings,
    generation: GroundedDraft,
) -> None:
    ticket = await tickets.add(make_ticket(auto_send_allowed=True))
    service = build_service(
        tickets=tickets,
        audit=audit,
        queue=queue,
        settings=settings,
        knowledge_base=PresetKnowledgeBase(retrieved()),
        llm=ScriptedLLM(generation),
    )

    result = await service.process(ticket.id)

    assert result is not None
    assert result.status == "awaiting_operator"
    details = await only_audit_record(audit, ticket.id)
    assert details["escalated"] is True
    assert details["degraded"] is False
    assert details["reason"] != ""


async def test_a_run_with_no_usable_context_is_escalated_without_a_generation_call(
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: CountingQueue,
    settings: Settings,
) -> None:
    ticket = await tickets.add(make_ticket(auto_send_allowed=True))
    llm = ScriptedLLM(grounded_draft())
    service = build_service(
        tickets=tickets,
        audit=audit,
        queue=queue,
        settings=settings,
        knowledge_base=PresetKnowledgeBase(),
        llm=llm,
    )

    result = await service.process(ticket.id)

    assert result is not None
    assert result.status == "awaiting_operator"
    assert result.draft is not None
    assert result.draft.sources == ()
    assert result.draft.is_grounded is False
    assert result.draft.confidence == 0.0
    assert llm.calls == 0

    details = await only_audit_record(audit, ticket.id)
    assert details["escalated"] is True
    assert details["reason"] != ""


# --- degradation ---------------------------------------------------------


async def test_an_unavailable_llm_degrades_to_a_draft_built_from_retrieval(
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: CountingQueue,
    settings: Settings,
) -> None:
    ticket = await tickets.add(make_ticket(auto_send_allowed=True))
    knowledge_base = PresetKnowledgeBase(retrieved())
    service = build_service(
        tickets=tickets,
        audit=audit,
        queue=queue,
        settings=settings,
        knowledge_base=knowledge_base,
        llm=ScriptedLLM(LLMUnavailableError("Gemini API is unavailable")),
    )

    result = await service.process(ticket.id)

    assert result is not None
    assert result.status == "awaiting_operator"
    assert result.draft is not None
    assert result.draft.degraded is True
    assert result.draft.is_grounded is False
    assert result.draft.confidence == 0.0
    assert result.draft.sources == (SOURCE,)
    assert result.draft.text == DEGRADED_WITH_CONTEXT.format(
        context=f"[{SOURCE}]\n{FRAGMENT_TEXT}"
    )
    # Retrieval is retried independently of the generator.
    assert knowledge_base.queries == [TICKET_TEXT, TICKET_TEXT]


async def test_a_degraded_run_is_audited_as_degraded_and_escalated(
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: CountingQueue,
    settings: Settings,
) -> None:
    ticket = await tickets.add(make_ticket(auto_send_allowed=True))
    service = build_service(
        tickets=tickets,
        audit=audit,
        queue=queue,
        settings=settings,
        knowledge_base=PresetKnowledgeBase(retrieved()),
        llm=ScriptedLLM(LLMUnavailableError("Gemini API is unavailable")),
    )

    await service.process(ticket.id)

    details = await only_audit_record(audit, ticket.id)
    assert details["degraded"] is True
    assert details["escalated"] is True
    assert details["status"] == "awaiting_operator"
    assert details["reason"] == "Gemini API is unavailable"
    assert details["sources"] == [SOURCE]


async def test_fragments_below_the_floor_are_not_handed_to_the_operator_either(
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: CountingQueue,
    settings: Settings,
) -> None:
    ticket = await tickets.add(make_ticket(auto_send_allowed=True))
    service = build_service(
        tickets=tickets,
        audit=audit,
        queue=queue,
        settings=settings,
        knowledge_base=PresetKnowledgeBase(retrieved(score=MIN_RETRIEVAL_SCORE - 0.01)),
        llm=ScriptedLLM(LLMUnavailableError("Gemini API is unavailable")),
    )

    result = await service.process(ticket.id)

    assert result is not None
    assert result.draft is not None
    # The LLM is scripted to fail, but it is never consulted: retrieval found
    # nothing above the floor, so the graph escalates first. The operator must
    # therefore be told the real reason and *not* that generation is down.
    assert result.draft.text == ESCALATION_NOTE.format(reason=_NO_CONTEXT_REASON)
    assert DEGRADED_WITHOUT_CONTEXT not in result.draft.text
    # The whole point of the floor: the weak fragment leaks neither as text nor
    # as a citation.
    assert FRAGMENT_TEXT not in result.draft.text
    assert result.draft.sources == ()


async def test_a_dead_llm_and_a_dead_index_still_produce_a_ticket_for_an_operator(
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: CountingQueue,
    settings: Settings,
) -> None:
    ticket = await tickets.add(make_ticket(auto_send_allowed=True))
    service = build_service(
        tickets=tickets,
        audit=audit,
        queue=queue,
        settings=settings,
        knowledge_base=PresetKnowledgeBase(
            fails_with=KnowledgeBaseUnavailableError("Knowledge base is not loaded")
        ),
        llm=ScriptedLLM(LLMUnavailableError("Gemini API is unavailable")),
    )

    result = await service.process(ticket.id)

    assert result is not None
    assert result.status == "awaiting_operator"
    assert result.draft is not None
    assert result.draft.text == DEGRADED_WITHOUT_CONTEXT
    assert result.draft.sources == ()
    assert result.draft.degraded is True

    details = await only_audit_record(audit, ticket.id)
    assert details["degraded"] is True
    assert details["reason"] == "Knowledge base is not loaded"


# --- tickets that must not be drafted for -------------------------------


async def test_processing_an_unknown_ticket_returns_none_without_raising(
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: CountingQueue,
    settings: Settings,
) -> None:
    service = build_service(
        tickets=tickets,
        audit=audit,
        queue=queue,
        settings=settings,
        knowledge_base=PresetKnowledgeBase(retrieved()),
        llm=ScriptedLLM(grounded_draft()),
    )

    assert await service.process(uuid4()) is None
    assert await audit.records() == []


async def test_processing_an_untriaged_ticket_leaves_it_untouched(
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: CountingQueue,
    settings: Settings,
) -> None:
    ticket = await tickets.add(
        make_ticket(triaged=False).model_copy(update={"status": "triaged"})
    )
    llm = ScriptedLLM(grounded_draft())
    service = build_service(
        tickets=tickets,
        audit=audit,
        queue=queue,
        settings=settings,
        knowledge_base=PresetKnowledgeBase(retrieved()),
        llm=llm,
    )

    assert await service.process(ticket.id) is None

    stored = await tickets.get(ticket.id)
    assert stored is not None
    assert stored.status == "triaged"
    assert stored.draft is None
    assert llm.calls == 0
    assert await audit.records() == []


# --- the queue hand-off --------------------------------------------------


async def test_process_next_takes_the_ticket_from_the_queue_and_acknowledges_it(
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: CountingQueue,
    settings: Settings,
) -> None:
    ticket = await tickets.add(make_ticket(auto_send_allowed=True))
    service = build_service(
        tickets=tickets,
        audit=audit,
        queue=queue,
        settings=settings,
        knowledge_base=PresetKnowledgeBase(retrieved()),
        llm=ScriptedLLM(grounded_draft()),
    )
    assert queue.submit(ticket.id) is True

    result = await service.process_next()

    assert result is not None and result.id == ticket.id
    assert result.status == "draft_ready"
    assert queue.pending() == 0
    assert queue.acknowledged == 1


@pytest.mark.parametrize(
    "known_ticket",
    [True, False],
    ids=["run_degrades", "ticket_is_unknown"],
)
async def test_process_next_acknowledges_the_item_even_when_the_run_does_not_produce_a_draft(
    tickets: AbstractTicketRepository,
    audit: AbstractAuditLog,
    queue: CountingQueue,
    settings: Settings,
    known_ticket: bool,
) -> None:
    ticket = await tickets.add(make_ticket(auto_send_allowed=True))
    service = build_service(
        tickets=tickets,
        audit=audit,
        queue=queue,
        settings=settings,
        knowledge_base=PresetKnowledgeBase(
            fails_with=KnowledgeBaseUnavailableError("Knowledge base is not loaded")
        ),
        llm=ScriptedLLM(LLMUnavailableError("Gemini API is unavailable")),
    )
    queue.submit(ticket.id if known_ticket else uuid4())

    result = await service.process_next()

    assert (result is not None) is known_ticket
    assert queue.pending() == 0
    assert queue.acknowledged == 1
