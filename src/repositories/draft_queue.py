"""
The hand-off between the synchronous triage path and the asynchronous draft path.

**This queue is not durable, and that is a known, deliberate simplification.**
It lives inside the process: if the container restarts or the worker crashes,
every ticket waiting here is lost, and nothing redelivers it. A PoC can accept
that; production cannot. The target design replaces this class with RabbitMQ —
durable queues, publisher confirms, per-message ack so a crashed consumer
redelivers instead of dropping, and a dead-letter queue for poison tickets. See
``docs/architecture.md`` and ``docs/risks-and-ops.md``.

What the PoC does do is make the gap visible rather than silent: the queue is
bounded, a full queue reports failure to the caller instead of blocking or
dropping, and the lifespan logs how many tickets were still pending at shutdown.
"""

import asyncio
from uuid import UUID


class DraftQueue:
    """Bounded FIFO of ticket ids awaiting draft generation."""

    def __init__(self, maxsize: int) -> None:
        self._queue: asyncio.Queue[UUID] = asyncio.Queue(maxsize=maxsize)

    def submit(self, ticket_id: UUID) -> bool:
        """Enqueue without waiting.

        Returns ``False`` when the queue is full — backpressure is surfaced to
        the caller, which routes the ticket to an operator rather than dropping
        it. Under an incident spike this is the behaviour that keeps the system
        honest: fewer drafts, no lost tickets.
        """
        try:
            self._queue.put_nowait(ticket_id)
        except asyncio.QueueFull:
            return False
        return True

    async def next(self) -> UUID:
        """Wait for the next ticket id."""
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    def pending(self) -> int:
        """How many tickets would be lost if the process died right now."""
        return self._queue.qsize()
