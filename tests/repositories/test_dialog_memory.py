"""Tests for InMemoryDialogMemory and its ChatMessage payload."""

import pytest

from src.models.domain import ChatMessage
from src.repositories.dialog_memory import InMemoryDialogMemory


def _user(text: str) -> ChatMessage:
    return ChatMessage(role="user", text=text)


@pytest.fixture
def memory() -> InMemoryDialogMemory:
    return InMemoryDialogMemory()


async def test_history_of_unknown_chat_is_empty(memory):
    assert await memory.history("chat-1") == []


async def test_history_returns_messages_oldest_first(memory):
    await memory.append("chat-1", _user("первое"))
    await memory.append("chat-1", ChatMessage(role="assistant", text="второе"))
    await memory.append("chat-1", _user("третье"))

    history = await memory.history("chat-1")

    assert [(m.role, m.text) for m in history] == [
        ("user", "первое"),
        ("assistant", "второе"),
        ("user", "третье"),
    ]


async def test_history_returns_a_copy_not_the_internal_list(memory):
    await memory.append("chat-1", _user("оригинал"))

    snapshot = await memory.history("chat-1")
    snapshot.append(_user("вмешательство"))
    snapshot.pop(0)

    assert [m.text for m in await memory.history("chat-1")] == ["оригинал"]


@pytest.mark.parametrize(
    ("max_messages", "appended", "expected_texts"),
    [
        (3, 5, ["m2", "m3", "m4"]),
        (1, 4, ["m3"]),
    ],
    ids=["keeps-last-three", "keeps-only-newest"],
)
async def test_append_evicts_oldest_beyond_max_messages(
    max_messages, appended, expected_texts
):
    memory = InMemoryDialogMemory(max_messages=max_messages)

    for i in range(appended):
        await memory.append("chat-1", _user(f"m{i}"))

    assert [m.text for m in await memory.history("chat-1")] == expected_texts


async def test_default_retention_is_ten_messages(memory):
    for i in range(12):
        await memory.append("chat-1", _user(f"m{i}"))

    history = await memory.history("chat-1")

    assert len(history) == 10
    assert history[0].text == "m2"
    assert history[-1].text == "m11"


async def test_chats_are_isolated_from_each_other(memory):
    await memory.append("chat-1", _user("для первого"))
    await memory.append("chat-2", _user("для второго"))

    assert [m.text for m in await memory.history("chat-1")] == ["для первого"]
    assert [m.text for m in await memory.history("chat-2")] == ["для второго"]


async def test_ping_is_true(memory):
    assert await memory.ping() is True


@pytest.mark.parametrize(
    ("role", "text", "expected"),
    [
        ("user", "когда отпуск?", "user: когда отпуск?"),
        ("assistant", "через 14 дней", "assistant: через 14 дней"),
    ],
    ids=["user", "assistant"],
)
def test_chat_message_as_line_formats_role_and_text(role, text, expected):
    assert ChatMessage(role=role, text=text).as_line() == expected
