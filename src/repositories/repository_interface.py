from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import UUID
from typing import Optional

from src.models.domain import Item


class AbstractRepository(ABC):
    """Async CRUD contract for stored items."""

    @abstractmethod
    async def list(self, *, limit: int = 50, offset: int = 0) -> Sequence[Item]:
        """Return a page of items, newest first."""

    @abstractmethod
    async def count(self) -> int:
        """Total number of stored items, ignoring pagination."""

    @abstractmethod
    async def get(self, item_id: UUID) -> Optional[Item]:
        """Return one item, or ``None`` when it does not exist."""

    @abstractmethod
    async def add(self, item: Item) -> Item:
        """Persist a new item and return the stored version."""

    @abstractmethod
    async def update(self, item_id: UUID, item: Item) -> Optional[Item]:
        """Replace an existing item; ``None`` when it does not exist."""

    @abstractmethod
    async def delete(self, item_id: UUID) -> bool:
        """Remove an item. ``True`` when something was actually removed."""

    @abstractmethod
    async def ping(self) -> bool:
        """Cheap liveness probe of the backing store, used by /health."""

    async def load(self) -> None:
        """Optional warm-up, called once at startup. No-op by default."""
