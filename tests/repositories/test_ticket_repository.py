"""The in-memory ticket store.

The interesting part of this class is not CRUD, it is ``add()``: ``external_id``
uniqueness is decided *here*, under the lock, because the caller's
check-then-act straddles an LLM call and cannot be atomic. A duplicate add
returns the ticket already stored and silently drops the argument, and the only
way a caller can tell is by comparing ids — so that is what these tests assert.
"""

from typing import Optional
from uuid import uuid4

import pytest

from src.models.domain import Channel, Ticket
from src.repositories.ticket_repository import InMemoryTicketRepository


def make_ticket(
    *,
    external_id: Optional[str] = None,
    text: str = "Как поменять способ оплаты?",
    channel: Channel = "chat",
) -> Ticket:
    return Ticket(id=uuid4(), channel=channel, text=text, external_id=external_id)


@pytest.fixture
def repository() -> InMemoryTicketRepository:
    return InMemoryTicketRepository()


# --- round trip ----------------------------------------------------------


async def test_add_returns_the_ticket_it_was_given(
    repository: InMemoryTicketRepository,
) -> None:
    ticket = make_ticket()

    added = await repository.add(ticket)

    assert added == ticket


async def test_a_stored_ticket_is_readable_back_by_id(
    repository: InMemoryTicketRepository,
) -> None:
    ticket = make_ticket()
    await repository.add(ticket)

    assert await repository.get(ticket.id) == ticket


async def test_get_returns_none_for_an_unknown_id(
    repository: InMemoryTicketRepository,
) -> None:
    await repository.add(make_ticket())

    assert await repository.get(uuid4()) is None


# --- the external_id index ----------------------------------------------


async def test_get_by_external_id_finds_the_indexed_ticket(
    repository: InMemoryTicketRepository,
) -> None:
    ticket = make_ticket(external_id="msg-42")
    await repository.add(ticket)

    assert await repository.get_by_external_id("msg-42") == ticket


async def test_get_by_external_id_returns_none_when_nothing_is_indexed(
    repository: InMemoryTicketRepository,
) -> None:
    await repository.add(make_ticket(external_id="msg-42"))

    assert await repository.get_by_external_id("msg-43") is None


async def test_a_ticket_added_without_an_external_id_is_not_indexed(
    repository: InMemoryTicketRepository,
) -> None:
    ticket = make_ticket(external_id=None)
    await repository.add(ticket)

    assert await repository.get(ticket.id) == ticket
    assert await repository.get_by_external_id("") is None


# --- uniqueness ----------------------------------------------------------


async def test_adding_a_duplicate_external_id_returns_the_ticket_already_stored(
    repository: InMemoryTicketRepository,
) -> None:
    """The caller detects the loss by comparing ids — nothing raises."""
    winner = await repository.add(make_ticket(external_id="msg-42", text="первый"))
    loser = make_ticket(external_id="msg-42", text="второй")

    returned = await repository.add(loser)

    assert returned == winner
    assert returned.id != loser.id


async def test_the_discarded_duplicate_is_not_retrievable_by_its_own_id(
    repository: InMemoryTicketRepository,
) -> None:
    await repository.add(make_ticket(external_id="msg-42", text="первый"))
    loser = make_ticket(external_id="msg-42", text="второй")

    await repository.add(loser)

    assert await repository.get(loser.id) is None


async def test_a_duplicate_add_leaves_exactly_one_ticket_stored(
    repository: InMemoryTicketRepository,
) -> None:
    winner = await repository.add(make_ticket(external_id="msg-42", text="первый"))

    await repository.add(make_ticket(external_id="msg-42", text="второй"))

    assert await repository.get(winner.id) == winner
    assert await repository.get_by_external_id("msg-42") == winner
    assert len(repository._tickets) == 1


async def test_different_external_ids_are_stored_side_by_side(
    repository: InMemoryTicketRepository,
) -> None:
    first = await repository.add(make_ticket(external_id="msg-1"))
    second = await repository.add(make_ticket(external_id="msg-2"))

    assert second.id != first.id
    assert await repository.get_by_external_id("msg-1") == first
    assert await repository.get_by_external_id("msg-2") == second


async def test_tickets_without_an_external_id_never_collide(
    repository: InMemoryTicketRepository,
) -> None:
    """``None`` is not an idempotency key: three anonymous tickets are three tickets."""
    tickets = [make_ticket(external_id=None) for _ in range(3)]

    added = [await repository.add(ticket) for ticket in tickets]

    assert added == tickets
    assert len({ticket.id for ticket in added}) == 3
    for ticket in tickets:
        assert await repository.get(ticket.id) == ticket


# --- save ----------------------------------------------------------------


async def test_save_replaces_the_stored_version(
    repository: InMemoryTicketRepository,
) -> None:
    ticket = await repository.add(make_ticket(text="исходный"))
    updated = ticket.model_copy(update={"status": "awaiting_operator"})

    saved = await repository.save(updated)

    assert saved == updated
    assert await repository.get(ticket.id) == updated


async def test_save_keeps_the_external_id_index_pointing_at_the_new_version(
    repository: InMemoryTicketRepository,
) -> None:
    """Triage saves a corrected ticket after storing it; a redelivery must see it."""
    ticket = await repository.add(make_ticket(external_id="msg-42"))
    updated = ticket.model_copy(update={"status": "awaiting_operator"})

    await repository.save(updated)

    assert await repository.get_by_external_id("msg-42") == updated


# --- lifecycle hooks -----------------------------------------------------


async def test_ping_reports_the_store_as_reachable(
    repository: InMemoryTicketRepository,
) -> None:
    assert await repository.ping() is True


async def test_load_is_a_no_op(repository: InMemoryTicketRepository) -> None:
    ticket = await repository.add(make_ticket(external_id="msg-42"))

    assert await repository.load() is None
    assert await repository.get(ticket.id) == ticket
    assert await repository.get_by_external_id("msg-42") == ticket
