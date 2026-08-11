"""
Topic and risk classification.

Two implementations behind one Protocol:

* :class:`LlmTopicClassifier` — what runs today. One structured Gemini call per
  ticket. Accurate out of the box and needs no labelled data, which is exactly
  why it is the right *first* step: there is no production traffic to learn from
  yet. It is also slow and metered, so it is the reason this PoC does not meet
  the 500 ms target (see ``docs/ml.md``).
* :class:`RuleTopicClassifier` — what runs when the provider is unreachable.
  Deterministic, sub-millisecond, and deliberately capped below the auto-send
  confidence threshold so a degraded decision can never close a ticket.

The Protocol is the seam the distilled model will slot into once there are
logged, labelled tickets to train it on: a third implementation, one changed
line in ``src/dependencies.py``, no other file touched.
"""

from typing import Protocol

from src.agent.llm import SupportsStructuredGeneration
from src.agent.prompts import CLASSIFIER_SYSTEM_PROMPT
from src.agent.schemas import TicketClassification
from src.ml.rules import evaluate
from src.utils import get_logger


logger = get_logger(__name__)

#: The rule classifier is a safety net, not a decision maker. Keeping its
#: confidence under ``Settings.min_topic_confidence`` (0.75) means every ticket
#: it classifies is routed to a human by construction.
_RULE_MATCH_CONFIDENCE = 0.6
_RULE_NO_MATCH_CONFIDENCE = 0.2


class TopicClassifier(Protocol):
    async def classify(self, text: str) -> TicketClassification: ...


class LlmTopicClassifier:
    """Classification by one structured LLM call."""

    def __init__(self, llm: SupportsStructuredGeneration) -> None:
        self._llm = llm

    async def classify(self, text: str) -> TicketClassification:
        """Raises :class:`LLMUnavailableError`; the caller falls back."""
        return await self._llm.generate_structured(
            system=CLASSIFIER_SYSTEM_PROMPT,
            user=f"Обращение пользователя:\n<<<\n{text}\n>>>",
            schema=TicketClassification,
        )


class RuleTopicClassifier:
    """Keyword fallback used when the LLM API is unavailable."""

    async def classify(self, text: str) -> TicketClassification:
        verdict = evaluate(text)
        if verdict.category is None:
            return TicketClassification(
                category="other",
                risk=verdict.risk,
                confidence=_RULE_NO_MATCH_CONFIDENCE,
                reason="LLM недоступен, правила не дали категорию — нужен оператор",
            )
        return TicketClassification(
            category=verdict.category,
            risk=verdict.risk,
            confidence=_RULE_MATCH_CONFIDENCE,
            reason=f"LLM недоступен, категория по правилам: {', '.join(verdict.hits)}",
        )
