"""
The hand-off queue between the synchronous and the asynchronous path.

The queue is deliberately bounded and deliberately non-blocking on submit: under
an incident spike the caller must learn that the draft path is saturated and
route the ticket to an operator, rather than block the HTTP handler or drop the
ticket silently. Those two properties are what this file pins down.
"""

from uuid import UUID, uuid4

import pytest

from src.repositories.draft_queue import DraftQueue


def ids(count: int) -> list[UUID]:
    return [uuid4() for _ in range(count)]


async def test_tickets_come_back_in_the_order_they_were_submitted() -> None:
    queue = DraftQueue(maxsize=10)
    submitted = ids(3)

    for ticket_id in submitted:
        queue.submit(ticket_id)

    assert [await queue.next() for _ in submitted] == submitted


async def test_submit_reports_success_while_there_is_room() -> None:
    queue = DraftQueue(maxsize=2)

    assert queue.submit(uuid4()) is True
    assert queue.submit(uuid4()) is True


async def test_submit_reports_failure_instead_of_blocking_when_the_queue_is_full() -> None:
    queue = DraftQueue(maxsize=2)
    accepted = ids(2)
    for ticket_id in accepted:
        queue.submit(ticket_id)

    rejected = uuid4()
    result = queue.submit(rejected)

    assert result is False
    assert queue.pending() == 2
    # The rejected id was not queued behind the accepted ones.
    assert [await queue.next() for _ in accepted] == accepted


async def test_a_full_queue_accepts_again_once_an_item_is_taken() -> None:
    queue = DraftQueue(maxsize=1)
    queue.submit(uuid4())
    assert queue.submit(uuid4()) is False

    await queue.next()

    assert queue.submit(uuid4()) is True


@pytest.mark.parametrize("submitted", [0, 1, 5], ids=["empty", "single", "several"])
async def test_pending_reports_how_many_tickets_would_be_lost(submitted: int) -> None:
    queue = DraftQueue(maxsize=10)

    for ticket_id in ids(submitted):
        queue.submit(ticket_id)

    assert queue.pending() == submitted


async def test_pending_drops_as_tickets_are_taken() -> None:
    queue = DraftQueue(maxsize=10)
    for ticket_id in ids(2):
        queue.submit(ticket_id)

    await queue.next()

    assert queue.pending() == 1
