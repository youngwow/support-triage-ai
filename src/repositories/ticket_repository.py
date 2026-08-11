"""
Ticket storage.

In-memory for the PoC. The contract below is what a real implementation would
have to satisfy — swapping in PostgreSQL means adding a class and changing
``get_ticket_repository()`` in ``src/dependencies.py``, nothing else.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from src.models.domain import Ticket


class AbstractTicketRepository(ABC):
    @abstractmethod
    async def add(self, ticket: Ticket) -> Ticket:
        """Store a new ticket and index it by ``external_id`` when present.

        ``external_id`` is unique: if one is already stored, the existing ticket
        is returned unchanged and the argument is discarded. Callers detect that
        by comparing ids. Uniqueness is decided here rather than by the caller
        because the caller's check-then-act cannot be atomic across the LLM call
        that sits between the two.
        """

    @abstractmethod
    async def get(self, ticket_id: UUID) -> Optional[Ticket]:
        """Return the ticket, or ``None`` when it is unknown."""

    @abstractmethod
    async def get_by_external_id(self, external_id: str) -> Optional[Ticket]:
        """Look up by the channel-side id — the idempotency key."""

    @abstractmethod
    async def save(self, ticket: Ticket) -> Ticket:
        """Replace a stored ticket with a newer version of itself."""

    @abstractmethod
    async def ping(self) -> bool:
        """Backs the readiness probe."""

    async def load(self) -> None:
        """Optional warm-up, awaited once from the app lifespan."""
        return None


class InMemoryTicketRepository(AbstractTicketRepository):
    """Process-local store. Everything is lost on restart — by design, for a PoC."""

    def __init__(self) -> None:
        self._tickets: dict[UUID, Ticket] = {}
        self._by_external_id: dict[str, UUID] = {}
        self._lock = asyncio.Lock()

    async def add(self, ticket: Ticket) -> Ticket:
        async with self._lock:
            if ticket.external_id is not None:
                existing_id = self._by_external_id.get(ticket.external_id)
                if existing_id is not None:
                    # A concurrent delivery of the same message won the race
                    # while this one was being classified.
                    return self._tickets[existing_id]
                self._by_external_id[ticket.external_id] = ticket.id
            self._tickets[ticket.id] = ticket
            return ticket

    async def get(self, ticket_id: UUID) -> Optional[Ticket]:
        async with self._lock:
            return self._tickets.get(ticket_id)

    async def get_by_external_id(self, external_id: str) -> Optional[Ticket]:
        async with self._lock:
            ticket_id = self._by_external_id.get(external_id)
            return None if ticket_id is None else self._tickets.get(ticket_id)

    async def save(self, ticket: Ticket) -> Ticket:
        async with self._lock:
            self._tickets[ticket.id] = ticket
            return ticket

    async def ping(self) -> bool:
        return True
