"""
The decision trail.

Every automatic decision the system makes writes exactly one record here, and
the metrics endpoint is computed by reading these back rather than by keeping a
parallel set of counters. One store, two views: a number on the dashboard can
always be traced to the records that produced it.

Append-only on purpose — an audit log you can edit is not an audit log.
"""

import asyncio
from abc import ABC, abstractmethod
from uuid import UUID

from src.models.domain import AuditRecord


class AbstractAuditLog(ABC):
    @abstractmethod
    async def append(self, record: AuditRecord) -> None:
        """Add one record. There is deliberately no update and no delete."""

    @abstractmethod
    async def for_ticket(self, ticket_id: UUID) -> list[AuditRecord]:
        """Every record for one ticket, oldest first."""

    @abstractmethod
    async def records(self) -> list[AuditRecord]:
        """The whole log, oldest first. Backs the metrics projection."""

    @abstractmethod
    async def ping(self) -> bool:
        """Backs the readiness probe."""

    async def load(self) -> None:
        """Optional warm-up, awaited once from the app lifespan."""
        return None


class InMemoryAuditLog(AbstractAuditLog):
    """Process-local log. A real deployment writes to append-only storage."""

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []
        self._lock = asyncio.Lock()

    async def append(self, record: AuditRecord) -> None:
        async with self._lock:
            self._records.append(record)

    async def for_ticket(self, ticket_id: UUID) -> list[AuditRecord]:
        async with self._lock:
            return [record for record in self._records if record.ticket_id == ticket_id]

    async def records(self) -> list[AuditRecord]:
        async with self._lock:
            return list(self._records)

    async def ping(self) -> bool:
        return True
