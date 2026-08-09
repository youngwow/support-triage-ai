from typing import Optional

from langgraph.graph.state import CompiledStateGraph

from src.agent.state import initial_state
from src.config import Settings
from src.exceptions import LLMUnavailableError
from src.models.domain import AssistantReply, ChatMessage
from src.repositories.dialog_memory import AbstractDialogMemory
from src.repositories.knowledge_base import AbstractKnowledgeBase
from src.utils import get_logger


logger = get_logger(__name__)


class AssistantService:
    """Runs the agent graph for one incoming message and keeps the dialog."""

    def __init__(
        self,
        *,
        graph: CompiledStateGraph,
        dialog_memory: AbstractDialogMemory,
        knowledge_base: AbstractKnowledgeBase,
        settings: Settings,
    ) -> None:
        self._graph = graph
        self._dialog_memory = dialog_memory
        self._kb = knowledge_base
        self._settings = settings

    async def handle_message(
        self, 
        *, 
        chat_id: str, 
        user_id: Optional[str], 
        text: str
    ) -> AssistantReply:
        history = await self._dialog_memory.history(chat_id)
        state = initial_state(
            user_id=user_id or chat_id, 
            question=text, 
            history=history
        )
        try:
            result = await self._graph.ainvoke(state)
            reply = AssistantReply(
                text=result["answer"] or "Извините, не удалось сформировать ответ.",
                route="escalate" if result["escalated"] else "knowledge_base",
                escalated=result["escalated"],
                escalation_reason=result["escalation_reason"],
                sources=result["sources"],
            )
        except LLMUnavailableError:
            logger.warning("LLM unavailable — serving a degraded keyword answer")
            reply = await self._degraded_reply(text)

        await self._dialog_memory.append(chat_id, ChatMessage(role="user", text=text))
        await self._dialog_memory.append(chat_id, ChatMessage(role="assistant", text=reply.text))
        return reply

    async def _degraded_reply(self, question: str) -> AssistantReply:
        """
        Graceful degradation: raw retrieval without summarisation.
        """
        try:
            results = await self._kb.search(question, k=self._settings.retrieval_top_k)
        except Exception as exc:
            logger.warning(f"Degraded-mode retrieval failed too: {exc}")
            results = []
        allowed = [
            result
            for result in results
            if result.chunk.source not in self._settings.restricted_sources
            and result.score >= self._settings.min_retrieval_score
        ]
        if not allowed:
            return AssistantReply(
                text=(
                    "Ассистент временно недоступен, и подходящего фрагмента "
                    "регламента найти не удалось. Пожалуйста, попробуйте позже "
                    "или обратитесь в поддержку напрямую."
                ),
                route="degraded",
            )
        top = allowed[0]
        return AssistantReply(
            text=(
                "ИИ-ассистент временно недоступен, показываю наиболее релевантный "
                f"фрагмент регламента ({top.chunk.source}):\n\n{top.chunk.text}\n\n"
                "(режим деградации: ответ без LLM)"
            ),
            route="degraded",
            sources=[top.chunk.source],
        )
