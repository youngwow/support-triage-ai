"""AssistantService: graph orchestration, dialog memory and degradation.

The graph is a recording fake returning a preset ``AgentState``-shaped result
dict; the dialog memory is the real ``InMemoryDialogMemory``; the knowledge
base is a preset fake so the degraded path never touches FAISS or the LLM.
"""

from collections.abc import Sequence
from typing import Optional

import pytest

from src.config import Settings, get_settings
from src.exceptions import LLMUnavailableError
from src.models.domain import ChatMessage, DocumentChunk, RetrievedChunk
from src.repositories.dialog_memory import InMemoryDialogMemory
from src.repositories.knowledge_base import AbstractKnowledgeBase
from src.services.assistant_service import AssistantService


FALLBACK_TEXT = "Извините, не удалось сформировать ответ."
DEGRADED_MARKER = "(режим деградации: ответ без LLM)"
DEGRADED_APOLOGY = (
    "Ассистент временно недоступен, и подходящего фрагмента "
    "регламента найти не удалось. Пожалуйста, попробуйте позже "
    "или обратитесь в поддержку напрямую."
)


class RecordingGraph:
    """Records every state ``ainvoke`` receives; returns or raises a preset."""

    def __init__(
        self, result: Optional[dict] = None, *, error: Optional[Exception] = None
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[dict] = []

    async def ainvoke(self, state: dict) -> dict:
        self.calls.append(state)
        if self._error is not None:
            raise self._error
        assert self._result is not None, "graph was invoked without a scripted result"
        return self._result


class PresetKnowledgeBase(AbstractKnowledgeBase):
    """Returns a fixed result list; records every (query, k) it was asked."""

    def __init__(self, results: Sequence[RetrievedChunk] = ()) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, *, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return list(self._results)

    async def count(self) -> int:
        return len(self._results)

    async def ping(self) -> bool:
        return True


class RaisingKnowledgeBase(PresetKnowledgeBase):
    async def search(self, query: str, *, k: int) -> list[RetrievedChunk]:
        raise RuntimeError("index is gone")


def graph_result(**overrides) -> dict:
    """A final ``AgentState`` as ``graph.ainvoke`` would return it."""
    result: dict = {
        "user_id": "user-1",
        "question": "вопрос",
        "history": [],
        "decision": None,
        "chunks": [],
        "tool_results": {},
        "generation": None,
        "answer": "Ответ по регламенту.",
        "sources": ["leave_policy.md"],
        "escalated": False,
        "escalation_reason": None,
    }
    result.update(overrides)
    return result


def retrieved(
    source: str, score: float, *, text: str = "фрагмент регламента", chunk_id: int = 0
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(id=chunk_id, source=source, text=text), score=score
    )


@pytest.fixture
def settings() -> Settings:
    return get_settings()


def make_service(
    *,
    graph: RecordingGraph,
    settings: Settings,
    memory: Optional[InMemoryDialogMemory] = None,
    kb: Optional[PresetKnowledgeBase] = None,
) -> AssistantService:
    return AssistantService(
        graph=graph,
        dialog_memory=memory or InMemoryDialogMemory(max_messages=10),
        knowledge_base=kb if kb is not None else PresetKnowledgeBase(),
        settings=settings,
    )


# -- result mapping ------------------------------------------------------------


async def test_maps_a_successful_graph_run_to_a_knowledge_base_reply(settings):
    graph = RecordingGraph(
        graph_result(answer="Предупредить за 14 дней.", sources=["leave_policy.md"])
    )
    service = make_service(graph=graph, settings=settings)

    reply = await service.handle_message(chat_id="chat-1", user_id="user-1", text="Отпуск?")

    assert reply.text == "Предупредить за 14 дней."
    assert reply.route == "knowledge_base"
    assert reply.escalated is False
    assert reply.escalation_reason is None
    assert reply.sources == ["leave_policy.md"]


async def test_maps_an_escalated_graph_run_to_the_escalate_route(settings):
    reason = "инцидент безопасности"
    graph = RecordingGraph(
        graph_result(
            answer="Передаю оператору.",
            escalated=True,
            escalation_reason=reason,
            sources=[],
        )
    )
    service = make_service(graph=graph, settings=settings)

    reply = await service.handle_message(chat_id="chat-1", user_id="user-1", text="Потерял ноутбук")

    assert reply.route == "escalate"
    assert reply.escalated is True
    assert reply.escalation_reason == reason
    assert reply.sources == []


async def test_empty_answer_is_replaced_by_the_fallback_text(settings):
    graph = RecordingGraph(graph_result(answer=None, sources=[]))
    service = make_service(graph=graph, settings=settings)

    reply = await service.handle_message(chat_id="chat-1", user_id="user-1", text="Вопрос")

    assert reply.text == FALLBACK_TEXT
    assert reply.route == "knowledge_base"


# -- initial state -------------------------------------------------------------


@pytest.mark.parametrize(
    ("user_id", "expected"),
    [("employee-7", "employee-7"), (None, "chat-42")],
    ids=["explicit-user-id", "falls-back-to-chat-id"],
)
async def test_state_user_id_falls_back_to_chat_id(settings, user_id, expected):
    graph = RecordingGraph(graph_result())
    service = make_service(graph=graph, settings=settings)

    await service.handle_message(chat_id="chat-42", user_id=user_id, text="Вопрос?")

    assert len(graph.calls) == 1
    state = graph.calls[0]
    assert state["user_id"] == expected
    assert state["question"] == "Вопрос?"


async def test_history_from_memory_reaches_the_graph(settings):
    memory = InMemoryDialogMemory(max_messages=10)
    seeded = [
        ChatMessage(role="user", text="Привет"),
        ChatMessage(role="assistant", text="Здравствуйте!"),
    ]
    for message in seeded:
        await memory.append("chat-1", message)
    graph = RecordingGraph(graph_result())
    service = make_service(graph=graph, settings=settings, memory=memory)

    await service.handle_message(chat_id="chat-1", user_id="user-1", text="Вопрос")

    assert graph.calls[0]["history"] == seeded


# -- dialog persistence --------------------------------------------------------


async def test_appends_user_and_assistant_turns_on_success(settings):
    memory = InMemoryDialogMemory(max_messages=10)
    graph = RecordingGraph(graph_result(answer="Ответ."))
    service = make_service(graph=graph, settings=settings, memory=memory)

    await service.handle_message(chat_id="chat-1", user_id="user-1", text="Вопрос")

    history = await memory.history("chat-1")
    assert [(message.role, message.text) for message in history] == [
        ("user", "Вопрос"),
        ("assistant", "Ответ."),
    ]


async def test_appends_both_turns_in_degraded_mode_too(settings):
    memory = InMemoryDialogMemory(max_messages=10)
    graph = RecordingGraph(error=LLMUnavailableError())
    service = make_service(
        graph=graph, settings=settings, memory=memory, kb=PresetKnowledgeBase()
    )

    reply = await service.handle_message(chat_id="chat-1", user_id="user-1", text="Вопрос")

    history = await memory.history("chat-1")
    assert [(message.role, message.text) for message in history] == [
        ("user", "Вопрос"),
        ("assistant", reply.text),
    ]


async def test_memory_trim_keeps_only_the_newest_messages(settings):
    memory = InMemoryDialogMemory(max_messages=3)
    for text in ("a", "b", "c"):
        await memory.append("chat-1", ChatMessage(role="user", text=text))
    graph = RecordingGraph(graph_result(answer="Ответ."))
    service = make_service(graph=graph, settings=settings, memory=memory)

    await service.handle_message(chat_id="chat-1", user_id="user-1", text="Вопрос")

    history = await memory.history("chat-1")
    assert [message.text for message in history] == ["c", "Вопрос", "Ответ."]


# -- degraded mode -------------------------------------------------------------


async def test_degraded_reply_quotes_the_top_allowed_chunk(settings):
    kb = PresetKnowledgeBase(
        [
            retrieved("salary_and_grades.md", 0.95, text="Оклад: секрет"),
            retrieved("leave_policy.md", 0.42, text="слабое совпадение", chunk_id=1),
            retrieved("hardware_policy.md", 0.71, text="Ноутбук выдаётся за 3 дня.", chunk_id=2),
        ]
    )
    graph = RecordingGraph(error=LLMUnavailableError())
    service = make_service(graph=graph, settings=settings, kb=kb)

    reply = await service.handle_message(chat_id="chat-1", user_id="user-1", text="Как заказать ноутбук?")

    assert reply.route == "degraded"
    assert reply.escalated is False
    assert reply.escalation_reason is None
    assert reply.sources == ["hardware_policy.md"]
    assert "hardware_policy.md" in reply.text
    assert "Ноутбук выдаётся за 3 дня." in reply.text
    assert DEGRADED_MARKER in reply.text
    # the restricted and the below-threshold chunks never leak into the reply
    assert "Оклад: секрет" not in reply.text
    assert "слабое совпадение" not in reply.text
    assert kb.calls == [("Как заказать ноутбук?", settings.retrieval_top_k)]


async def test_degraded_score_threshold_is_inclusive(settings):
    boundary = retrieved("leave_policy.md", settings.min_retrieval_score, text="Ровно на пороге.")
    graph = RecordingGraph(error=LLMUnavailableError())
    service = make_service(graph=graph, settings=settings, kb=PresetKnowledgeBase([boundary]))

    reply = await service.handle_message(chat_id="chat-1", user_id="user-1", text="Вопрос")

    assert reply.sources == ["leave_policy.md"]
    assert "Ровно на пороге." in reply.text


async def test_degraded_apologises_when_no_chunk_is_usable(settings):
    kb = PresetKnowledgeBase(
        [
            retrieved("salary_and_grades.md", 0.95, text="Оклад: секрет"),
            retrieved("leave_policy.md", 0.2, text="мимо", chunk_id=1),
        ]
    )
    graph = RecordingGraph(error=LLMUnavailableError())
    service = make_service(graph=graph, settings=settings, kb=kb)

    reply = await service.handle_message(chat_id="chat-1", user_id="user-1", text="Вопрос")

    assert reply.text == DEGRADED_APOLOGY
    assert reply.route == "degraded"
    assert reply.sources == []


async def test_degraded_survives_a_search_failure(settings):
    memory = InMemoryDialogMemory(max_messages=10)
    graph = RecordingGraph(error=LLMUnavailableError())
    service = make_service(
        graph=graph, settings=settings, memory=memory, kb=RaisingKnowledgeBase()
    )

    reply = await service.handle_message(chat_id="chat-1", user_id="user-1", text="Вопрос")

    assert reply.text == DEGRADED_APOLOGY
    assert reply.route == "degraded"
    assert reply.sources == []
    # the failure did not stop the dialog from being persisted
    assert len(await memory.history("chat-1")) == 2
