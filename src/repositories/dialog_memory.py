import asyncio
from abc import ABC, abstractmethod

from src.models.domain import ChatMessage


class AbstractDialogMemory(ABC):
    """Per-chat message history for agent context and escalation dumps."""

    @abstractmethod
    async def history(self, chat_id: str) -> list[ChatMessage]:
        """Messages of one chat, oldest first."""

    @abstractmethod
    async def append(self, chat_id: str, message: ChatMessage) -> None:
        """Store one message, evicting the oldest beyond the retention limit."""

    @abstractmethod
    async def ping(self) -> bool:
        """True when the store can be reached."""

    async def load(self) -> None:
        """Optional warm-up hook awaited once from the app lifespan."""
        return None


class InMemoryDialogMemory(AbstractDialogMemory):
    def __init__(self, *, max_messages: int = 10) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self._storage: dict[str, list[ChatMessage]] = {}
        self._lock = asyncio.Lock()
        self._max_messages = max_messages

    async def history(self, chat_id: str) -> list[ChatMessage]:
        async with self._lock:
            return list(self._storage.get(chat_id, []))

    async def append(self, chat_id: str, message: ChatMessage) -> None:
        async with self._lock:
            messages = self._storage.setdefault(chat_id, [])
            messages.append(message)
            del messages[: -self._max_messages]

    async def ping(self) -> bool:
        return True
