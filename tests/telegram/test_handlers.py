"""aiogram handlers called directly — they are plain async functions.

The message stub only needs the attributes the handlers read (``text``,
``chat.id``, ``from_user.id``) plus an async ``answer`` recorder; no real
aiogram ``Message`` (and no bot session) is ever constructed.
"""

from types import SimpleNamespace
from typing import Optional

from src.models.domain import AssistantReply
from src.telegram.handlers import _GREETING, create_dispatcher, on_message, on_start


class FakeMessage:
    def __init__(
        self,
        *,
        text: Optional[str],
        chat_id: int = 42,
        from_user_id: Optional[int] = 7,
    ) -> None:
        self.text = text
        self.chat = SimpleNamespace(id=chat_id)
        self.from_user = (
            SimpleNamespace(id=from_user_id) if from_user_id is not None else None
        )
        self.answers: list[str] = []

    async def answer(self, text: str) -> None:
        self.answers.append(text)


class StubAssistant:
    def __init__(self, reply_text: str = "Ответ ассистента.") -> None:
        self.reply = AssistantReply(text=reply_text, route="knowledge_base")
        self.calls: list[dict] = []

    async def handle_message(self, *, chat_id, user_id, text) -> AssistantReply:
        self.calls.append({"chat_id": chat_id, "user_id": user_id, "text": text})
        return self.reply


async def test_start_command_answers_the_greeting():
    message = FakeMessage(text="/start")

    await on_start(message)

    assert message.answers == [_GREETING]


async def test_text_message_is_routed_to_the_assistant_and_answered():
    assistant = StubAssistant(reply_text="Предупредить за 14 дней.")
    message = FakeMessage(text="Отпуск?", chat_id=42, from_user_id=7)

    await on_message(message, assistant)

    assert assistant.calls == [{"chat_id": "42", "user_id": "7", "text": "Отпуск?"}]
    assert message.answers == ["Предупредить за 14 дней."]


async def test_missing_from_user_passes_user_id_none():
    assistant = StubAssistant()
    message = FakeMessage(text="Вопрос", from_user_id=None)

    await on_message(message, assistant)

    assert assistant.calls == [{"chat_id": "42", "user_id": None, "text": "Вопрос"}]


async def test_non_text_message_gets_the_fallback_and_skips_the_assistant():
    assistant = StubAssistant()
    message = FakeMessage(text=None)

    await on_message(message, assistant)

    assert message.answers == ["Пожалуйста, отправьте текстовое сообщение."]
    assert assistant.calls == []


def test_create_dispatcher_injects_the_assistant_into_workflow_data():
    assistant = StubAssistant()

    dispatcher = create_dispatcher(assistant)

    assert dispatcher["assistant"] is assistant
