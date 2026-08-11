"""
Nodes of the draft graph.

They are methods on a class constructed with its collaborators, so the graph is
pure logic over abstractions: a test builds the same graph with a scripted LLM
and a preset knowledge base and never touches a network.

Every node returns a *partial* state dict. Routers are synchronous and return a
label the path map resolves to a node name.
"""

from src.agent.prompts import GENERATOR_SYSTEM_PROMPT, format_context
from src.agent.schemas import DraftState, GroundedDraft
from src.config import Settings
from src.agent.llm import SupportsStructuredGeneration
from src.repositories.knowledge_base import AbstractKnowledgeBase
from src.utils import get_logger


logger = get_logger(__name__)

_NO_CONTEXT_REASON = "В базе знаний нет фрагментов, релевантных обращению"
_UNGROUNDED_REASON = "Черновик не подтверждён источниками"
_LOW_CONFIDENCE_REASON = "Низкая уверенность генератора ({confidence:.2f})"


class DraftNodes:
    def __init__(
        self,
        *,
        knowledge_base: AbstractKnowledgeBase,
        llm: SupportsStructuredGeneration,
        settings: Settings,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._llm = llm
        self._settings = settings

    async def retrieve(self, state: DraftState) -> dict:
        """Fetch candidate fragments and drop everything below the score floor.

        The floor matters more than the ranking: an irrelevant fragment that
        still comes back first is exactly what makes a generator hallucinate
        confidently.
        """
        found = await self._knowledge_base.search(
            state["question"], k=self._settings.retrieval_top_k
        )
        relevant = [
            chunk for chunk in found if chunk.score >= self._settings.min_retrieval_score
        ]
        return {"chunks": relevant}

    def route_after_retrieve(self, state: DraftState) -> str:
        return "generate" if state["chunks"] else "escalate"

    async def generate(self, state: DraftState) -> dict:
        """One structured call: the draft and the model's own verdict on it."""
        context = format_context(
            [(chunk.chunk.source, chunk.chunk.text) for chunk in state["chunks"]]
        )
        draft: GroundedDraft = await self._llm.generate_structured(
            system=GENERATOR_SYSTEM_PROMPT,
            user=(
                f"Категория обращения: {state['category']}\n"
                f"Уровень риска: {state['risk']}\n\n"
                f"Фрагменты базы знаний:\n{context}\n\n"
                f"Обращение пользователя:\n<<<\n{state['question']}\n>>>"
            ),
            schema=GroundedDraft,
        )
        return {
            "generation": draft,
            "answer": draft.answer,
            "sources": list(draft.sources),
        }

    def route_after_generate(self, state: DraftState) -> str:
        draft = state["generation"]
        if draft is None or not draft.is_grounded:
            return "escalate"
        if draft.confidence < self._settings.min_draft_confidence:
            return "escalate"
        return "end"

    async def escalate(self, state: DraftState) -> dict:
        """Hand the ticket to a human, keeping whatever context was gathered."""
        draft = state["generation"]
        if draft is None:
            reason = _NO_CONTEXT_REASON
        elif not draft.is_grounded:
            reason = _UNGROUNDED_REASON
        else:
            reason = _LOW_CONFIDENCE_REASON.format(confidence=draft.confidence)

        logger.info(f"Ticket {state['ticket_id']} escalated: {reason}")
        return {"escalated": True, "escalation_reason": reason}
