"""
The synchronous path: what a ticket is, how risky it is, and where it goes.

Order is load-bearing:

1. **Mask PII first.** Everything after this point — the stored ticket, the
   classifier prompt, the audit record — sees masked text only.
2. **Rules second.** They set a risk floor and can identify the topic on their
   own. Running them before the model means the expensive call never gets the
   chance to soften a decision the rules already made.
3. **Model third**, and its failure is not the request's failure: an unreachable
   provider falls back to the rule classifier, whose confidence is capped below
   the auto-send threshold, so a degraded ticket always reaches a human.

The whole thing is timed and the timing is stored. The case asks for < 500 ms
here and an LLM call does not fit in that budget — see ``docs/ml.md`` for why
that trade was made deliberately and what replaces it.
"""

from time import perf_counter
from typing import Optional
from uuid import UUID, uuid4

from src.agent.schemas import TicketClassification
from src.config import Settings
from src.exceptions import LLMUnavailableError
from src.ml.classifier import TopicClassifier
from src.ml.pii import mask_pii
from src.ml.rules import evaluate
from src.models.domain import (
    AuditRecord,
    Channel,
    Route,
    Ticket,
    TicketStatus,
    TriageDecision,
    max_risk,
)
from src.repositories.audit_log import AbstractAuditLog
from src.repositories.draft_queue import DraftQueue
from src.repositories.ticket_repository import AbstractTicketRepository
from src.utils import get_logger


logger = get_logger(__name__)


class TriageService:
    def __init__(
        self,
        *,
        tickets: AbstractTicketRepository,
        audit: AbstractAuditLog,
        classifier: TopicClassifier,
        fallback_classifier: TopicClassifier,
        queue: DraftQueue,
        settings: Settings,
    ) -> None:
        self._tickets = tickets
        self._audit = audit
        self._classifier = classifier
        self._fallback = fallback_classifier
        self._queue = queue
        self._settings = settings

    async def triage(
        self,
        *,
        channel: Channel,
        text: str,
        external_id: Optional[str] = None,
    ) -> tuple[Ticket, bool]:
        """Classify, route and store one ticket.

        Returns ``(ticket, created)``. ``created`` is ``False`` when
        ``external_id`` matched an existing ticket, so a redelivered message
        costs nothing and cannot be triaged twice.
        """
        if external_id is not None:
            existing = await self._tickets.get_by_external_id(external_id)
            if existing is not None:
                await self._audit.append(
                    AuditRecord(
                        ticket_id=existing.id,
                        event="deduplicated",
                        details={"external_id": external_id},
                    )
                )
                return existing, False

        started = perf_counter()

        masked = mask_pii(text)
        verdict = evaluate(masked.text)

        classifier_used = "llm"
        try:
            classification = await self._classifier.classify(masked.text)
        except LLMUnavailableError as exc:
            classifier_used = "rules"
            logger.warning(f"Classification degraded to rules: {exc.detail}")
            classification = await self._fallback.classify(masked.text)

        risk = max_risk(verdict.risk, classification.risk)
        if masked.types:
            # Personal data in the body means a human reads it, whatever the
            # topic turned out to be.
            risk = max_risk(risk, "medium")

        confidence = classification.confidence
        confident = confidence >= self._settings.min_topic_confidence

        route = self._route(
            risk=risk,
            confident=confident,
            injection=verdict.injection,
            degraded=classifier_used == "rules",
        )
        auto_send_allowed = (
            route == "auto_answer"
            and risk == "low"
            and confident
            and not masked.types
            and not verdict.injection
            and classifier_used == "llm"
        )

        latency_ms = (perf_counter() - started) * 1000.0

        decision = TriageDecision(
            category=classification.category,
            category_confidence=confidence,
            risk=risk,
            route=route,
            auto_send_allowed=auto_send_allowed,
            classifier=classifier_used,
            rule_hits=verdict.hits,
            pii_types=masked.types,
            injection_flagged=verdict.injection,
            reason=classification.reason,
            latency_ms=latency_ms,
        )

        ticket = Ticket(
            id=uuid4(),
            channel=channel,
            text=masked.text,
            external_id=external_id,
            status="triaged",
        )
        wants_draft = self._wants_draft(decision)
        status: TicketStatus = "queued" if wants_draft else "awaiting_operator"
        ticket = ticket.with_decision(decision, status=status)

        # Store before enqueueing, never the other way round: the worker looks
        # the ticket up by id, and a queue that hands out ids the repository has
        # not seen yet loses drafts under concurrency.
        stored = await self._tickets.add(ticket)
        if stored.id != ticket.id:
            # The store rejected this one: a concurrent delivery of the same
            # external_id got there first. Classification was paid for, but only
            # one ticket exists and only one draft will be generated.
            await self._audit.append(
                AuditRecord(
                    ticket_id=stored.id,
                    event="deduplicated",
                    details={"external_id": external_id, "concurrent": True},
                )
            )
            return stored, False
        ticket = stored

        if wants_draft and not self._queue.submit(ticket.id):
            # Backpressure: no draft for this one, but the ticket is not lost.
            # Correct the decision before it reaches the audit log — a trail
            # that records `auto_answer` for a ticket that went to a human is
            # worse than no trail at all, and the metrics read the same records.
            wants_draft = False
            decision = decision.model_copy(
                update={
                    "route": "operator_queue",
                    "auto_send_allowed": False,
                    "reason": (
                        f"{decision.reason} | Очередь черновиков переполнена, "
                        "тикет передан оператору без черновика"
                    ).lstrip(" |"),
                }
            )
            ticket = ticket.with_decision(decision, status="awaiting_operator")
            await self._tickets.save(ticket)
            await self._audit.append(
                AuditRecord(
                    ticket_id=ticket.id,
                    event="queue_rejected",
                    details={"pending": self._queue.pending()},
                )
            )

        if classifier_used == "rules":
            await self._audit.append(
                AuditRecord(
                    ticket_id=ticket.id,
                    event="llm_classification_failed",
                    details={"fallback": "rules"},
                )
            )
        await self._audit.append(
            AuditRecord(
                ticket_id=ticket.id,
                event="triaged",
                details={
                    "category": decision.category,
                    "category_confidence": decision.category_confidence,
                    "risk": decision.risk,
                    "route": decision.route,
                    "auto_send_allowed": decision.auto_send_allowed,
                    "classifier": decision.classifier,
                    "rule_hits": list(decision.rule_hits),
                    "pii_types": list(decision.pii_types),
                    "injection_flagged": decision.injection_flagged,
                    "latency_ms": decision.latency_ms,
                    "reason": decision.reason,
                },
            )
        )
        if wants_draft:
            await self._audit.append(
                AuditRecord(
                    ticket_id=ticket.id,
                    event="queued",
                    details={"pending": self._queue.pending()},
                )
            )

        return ticket, True

    async def get(self, ticket_id: UUID) -> Optional[Ticket]:
        return await self._tickets.get(ticket_id)

    async def audit_trail(self, ticket_id: UUID) -> list[AuditRecord]:
        return await self._audit.for_ticket(ticket_id)

    @staticmethod
    def _route(*, risk: str, confident: bool, injection: bool, degraded: bool) -> Route:
        """Risk decides first; uncertainty and degradation both mean a human."""
        if risk == "high" or injection:
            return "senior_escalation"
        if risk == "medium" or not confident or degraded:
            return "operator_queue"
        return "auto_answer"

    @staticmethod
    def _wants_draft(decision: TriageDecision) -> bool:
        """Whether it is worth spending a generation call on this ticket.

        Senior escalations never get one: a disputed charge or a legal threat is
        answered by a person, so the call would be pure cost. Injection-flagged
        text is never forwarded to the provider at all.
        """
        if decision.injection_flagged:
            return False
        return decision.route in ("auto_answer", "operator_queue")
