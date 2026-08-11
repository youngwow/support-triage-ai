"""
The asynchronous path: turn a triaged ticket into a suggested reply.

Nothing here ever sends anything to a user. A finished draft is either
``draft_ready`` — it passed every gate and in a production system would be the
candidate for automatic sending — or it is attached to a ticket that is
``awaiting_operator``, which is the outcome for anything risky, unconfident,
ungrounded or degraded.

Failure of the LLM or of retrieval is not an error here. It is a *route*: the
ticket gets whatever context could still be gathered and goes to a human.
"""

from typing import Optional
from uuid import UUID

from src.agent.prompts import (
    DEGRADED_WITH_CONTEXT,
    DEGRADED_WITHOUT_CONTEXT,
    ESCALATION_NOTE,
    format_context,
)
from src.agent.schemas import initial_draft_state
from src.config import Settings
from src.exceptions import AppError
from src.models.domain import AuditRecord, DraftAnswer, Ticket, TicketStatus
from src.repositories.audit_log import AbstractAuditLog
from src.repositories.draft_queue import DraftQueue
from src.repositories.knowledge_base import AbstractKnowledgeBase
from src.repositories.ticket_repository import AbstractTicketRepository
from src.utils import get_logger


logger = get_logger(__name__)


class DraftService:
    def __init__(
        self,
        *,
        tickets: AbstractTicketRepository,
        audit: AbstractAuditLog,
        knowledge_base: AbstractKnowledgeBase,
        queue: DraftQueue,
        graph,
        settings: Settings,
    ) -> None:
        self._tickets = tickets
        self._audit = audit
        self._knowledge_base = knowledge_base
        self._queue = queue
        self._graph = graph
        self._settings = settings

    async def process_next(self) -> Optional[Ticket]:
        """Take one ticket off the queue and draft for it.

        Tests call this directly instead of racing the background worker.
        """
        ticket_id = await self._queue.next()
        try:
            return await self.process(ticket_id)
        finally:
            self._queue.task_done()

    async def process(self, ticket_id: UUID) -> Optional[Ticket]:
        ticket = await self._tickets.get(ticket_id)
        if ticket is None or ticket.decision is None:
            logger.warning(f"Draft requested for unknown or untriaged ticket {ticket_id}")
            return None

        degraded = False
        escalated = False
        reason: Optional[str] = None

        try:
            result = await self._graph.ainvoke(
                initial_draft_state(
                    ticket_id=str(ticket.id),
                    question=ticket.text,
                    category=ticket.decision.category,
                    risk=ticket.decision.risk,
                )
            )
        except AppError as exc:
            # The provider or the index is down. Degrade, never 500 a worker.
            logger.warning(f"Draft degraded for {ticket.id}: {exc.detail}")
            draft = await self._degraded_draft(ticket)
            degraded = True
            escalated = True
            reason = exc.detail
        else:
            generation = result["generation"]
            escalated = bool(result["escalated"])
            reason = result["escalation_reason"]
            answer = result["answer"]
            if answer is None:
                # Escalated before anything was generated — retrieval found
                # nothing usable. Nothing was *down*, so the operator must not
                # be told the generation service is unavailable; give them the
                # real reason instead.
                answer = ESCALATION_NOTE.format(
                    reason=reason or "требуется решение оператора"
                )
            draft = DraftAnswer(
                text=answer,
                sources=tuple(result["sources"]),
                is_grounded=bool(generation is not None and generation.is_grounded),
                confidence=generation.confidence if generation is not None else 0.0,
                degraded=False,
            )

        status: TicketStatus = (
            "draft_ready"
            if not escalated and ticket.decision.auto_send_allowed
            else "awaiting_operator"
        )
        ticket = ticket.with_draft(draft, status=status)
        await self._tickets.save(ticket)

        await self._audit.append(
            AuditRecord(
                ticket_id=ticket.id,
                event="drafted",
                details={
                    "status": status,
                    "degraded": degraded,
                    "escalated": escalated,
                    "is_grounded": draft.is_grounded,
                    "confidence": draft.confidence,
                    "sources": list(draft.sources),
                    "reason": reason or "",
                },
            )
        )
        return ticket

    async def _degraded_draft(self, ticket: Ticket) -> DraftAnswer:
        """No generation available — hand the operator the raw source material.

        Retrieval is tried independently: the LLM being down does not mean the
        index is, and a relevant policy fragment is worth more to an operator
        than an apology.
        """
        try:
            found = await self._knowledge_base.search(
                ticket.text, k=self._settings.retrieval_top_k
            )
            relevant = [
                chunk
                for chunk in found
                if chunk.score >= self._settings.min_retrieval_score
            ]
        except AppError as exc:
            logger.warning(f"Degraded retrieval unavailable for {ticket.id}: {exc.detail}")
            relevant = []

        if not relevant:
            return DraftAnswer(
                text=DEGRADED_WITHOUT_CONTEXT,
                sources=(),
                is_grounded=False,
                confidence=0.0,
                degraded=True,
            )

        context = format_context(
            [(chunk.chunk.source, chunk.chunk.text) for chunk in relevant]
        )
        return DraftAnswer(
            text=DEGRADED_WITH_CONTEXT.format(context=context),
            sources=tuple(dict.fromkeys(chunk.chunk.source for chunk in relevant)),
            is_grounded=False,
            confidence=0.0,
            degraded=True,
        )
