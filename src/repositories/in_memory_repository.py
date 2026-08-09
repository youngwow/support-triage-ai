import asyncio
from collections.abc import Sequence
from uuid import UUID
from typing import Optional

from src.models.domain import Item
from src.repositories.repository_interface import AbstractRepository


class InMemoryRepository(AbstractRepository):
    def __init__(self) -> None:
        self._storage: dict[UUID, Item] = {}
        self._lock = asyncio.Lock()

    async def list(self, *, limit: int = 50, offset: int = 0) -> Sequence[Item]:
        async with self._lock:
            items = list(self._storage.values())
        return items[::-1][offset : offset + limit]

    async def count(self) -> int:
        async with self._lock:
            return len(self._storage)

    async def get(self, item_id: UUID) -> Optional[Item]:
        async with self._lock:
            return self._storage.get(item_id)

    async def add(self, item: Item) -> Item:
        async with self._lock:
            if item.id in self._storage:
                raise KeyError(item.id)
            self._storage[item.id] = item
        return item

    async def update(self, item_id: UUID, item: Item) -> Optional[Item]:
        async with self._lock:
            if item_id not in self._storage:
                return None
            if item.id != item_id:  # identity changed: re-key
                del self._storage[item_id]
            self._storage[item.id] = item
        return item

    async def delete(self, item_id: UUID) -> bool:
        async with self._lock:
            return self._storage.pop(item_id, None) is not None

    async def ping(self) -> bool:
        return True
