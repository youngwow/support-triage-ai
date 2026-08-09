from src.agent.llm import SupportsStructuredGeneration
from src.agent.prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    ESCALATION_REPLY,
    GENERATOR_SYSTEM_PROMPT,
    format_escalation,
)
from src.agent.schemas import GroundedAnswer, RouteDecision
from src.agent.state import AgentState
from src.agent.tools import AgentTools
from src.config import Settings
from src.models.domain import ChatMessage
from src.repositories.knowledge_base import AbstractKnowledgeBase
from src.utils import get_logger


logger = get_logger(__name__)

_HISTORY_CONTEXT_MESSAGES = 6


class AgentNodes:
    def __init__(
        self,
        *,
        knowledge_base: AbstractKnowledgeBase,
        tools: AgentTools,
        llm: SupportsStructuredGeneration,
        settings: Settings,
    ) -> None:
        self._kb = knowledge_base
        self._tools = tools
        self._llm = llm
        self._settings = settings

    async def classify(self, state: AgentState) -> dict:
        user = (
            f"История диалога:\n{self._history_text(state)}\n\n"
            f"Вопрос сотрудника: {state['question']}"
        )
        decision = await self._llm.generate_structured(
            system=CLASSIFIER_SYSTEM_PROMPT, user=user, schema=RouteDecision
        )
        return {"decision": decision}

    async def call_tools(self, state: AgentState) -> dict:
        decision = state["decision"]
        assert decision is not None  # routing guarantees classify ran
        results: dict[str, int] = {}
        if decision.needs_vacation_balance:
            results["vacation_balance_days"] = await self._tools.get_vacation_balance(
                state["user_id"]
            )
        if decision.needs_user_grade:
            results["grade"] = await self._tools.get_user_grade(state["user_id"])
        return {"tool_results": results}

    async def retrieve(self, state: AgentState) -> dict:
        results = await self._kb.search(
            state["question"], k=self._settings.retrieval_top_k
        )
        allowed = [
            result
            for result in results
            if result.chunk.source not in self._settings.restricted_sources
        ]
        return {"chunks": allowed}

    async def generate(self, state: AgentState) -> dict:
        context = "\n\n".join(
            f"[{result.chunk.source}] {result.chunk.text}" for result in state["chunks"]
        )
        tool_lines = (
            "\n".join(f"- {name}: {value}" for name, value in state["tool_results"].items())
            or "(не запрашивались)"
        )
        user = (
            f"Фрагменты регламентов:\n{context}\n\n"
            f"Данные сотрудника из HR-системы:\n{tool_lines}\n\n"
            f"История диалога:\n{self._history_text(state)}\n\n"
            f"Вопрос сотрудника: {state['question']}"
        )
        generation = await self._llm.generate_structured(
            system=GENERATOR_SYSTEM_PROMPT, user=user, schema=GroundedAnswer
        )
        return {
            "generation": generation,
            "answer": generation.answer,
            "sources": generation.sources,
        }

    async def escalate(self, state: AgentState) -> dict:
        reason = self._escalation_reason(state)
        dump = state["history"] + [ChatMessage(role="user", text=state["question"])]
        line = format_escalation(reason, dump)
        print(line)  # for a real operator queue
        logger.warning(line)
        return {
            "answer": ESCALATION_REPLY,
            "sources": [],
            "escalated": True,
            "escalation_reason": reason,
        }

    def route_after_classify(self, state: AgentState) -> str:
        decision = state["decision"]
        assert decision is not None
        if decision.route == "escalate":
            return "escalate"
        if decision.confidence < self._settings.min_route_confidence:
            return "escalate"
        if decision.needs_vacation_balance or decision.needs_user_grade:
            return "tools"
        return "retrieve"

    def route_after_retrieve(self, state: AgentState) -> str:
        chunks = state["chunks"]
        if not chunks:
            return "escalate"
        if max(result.score for result in chunks) < self._settings.min_retrieval_score:
            return "escalate"
        return "generate"

    def route_after_generate(self, state: AgentState) -> str:
        generation = state["generation"]
        assert generation is not None
        cites_restricted = any(
            source in self._settings.restricted_sources for source in generation.sources
        )
        if (
            not generation.is_grounded
            or generation.confidence < self._settings.min_route_confidence
            or cites_restricted
        ):
            return "escalate"
        return "end"

    def _history_text(self, state: AgentState) -> str:
        recent = state["history"][-_HISTORY_CONTEXT_MESSAGES:]
        return "\n".join(message.as_line() for message in recent) or "(пусто)"

    def _escalation_reason(self, state: AgentState) -> str:
        decision = state["decision"]
        if decision is None:
            return "сбой классификации запроса"
        if decision.route == "escalate":
            return decision.escalation_reason or "критичный или внерегламентный запрос"
        if decision.confidence < self._settings.min_route_confidence:
            return "низкая уверенность в классификации запроса"
        generation = state["generation"]
        if generation is None:
            return "в базе знаний не найдено релевантного ответа"
        if any(
            source in self._settings.restricted_sources for source in generation.sources
        ):
            return "ответ ссылается на конфиденциальные документы"
        return "ответ не подтверждён базой знаний"
