"""Coverage for ``src/repositories/in_memory_repository.py``.

Every test here is really a test of the ``AbstractRepository`` contract — when a
real backing store lands, these should be reusable against it.
"""

import asyncio
from uuid import UUID, uuid4

from src.models.domain import Item
from src.repositories.in_memory_repository import InMemoryRepository


def make_item(name: str = "item", item_id: UUID | None = None) -> Item:
    return Item(id=item_id or uuid4(), name=name)


async def test_add_then_get_round_trips_the_item() -> None:
    repository = InMemoryRepository()
    item = make_item()

    assert await repository.add(item) == item
    assert await repository.get(item.id) == item


async def test_get_returns_none_for_an_unknown_id() -> None:
    assert await InMemoryRepository().get(uuid4()) is None


async def test_add_rejects_a_duplicate_id() -> None:
    """A KeyError is the contract the service translates into 409."""
    repository = InMemoryRepository()
    item = make_item()
    await repository.add(item)

    try:
        await repository.add(item)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError on a duplicate id")


async def test_list_returns_newest_first() -> None:
    repository = InMemoryRepository()
    for name in ["first", "second", "third"]:
        await repository.add(make_item(name))

    items = await repository.list()

    assert [item.name for item in items] == ["third", "second", "first"]


async def test_list_applies_limit_and_offset() -> None:
    repository = InMemoryRepository()
    for name in ["a", "b", "c", "d"]:
        await repository.add(make_item(name))

    items = await repository.list(limit=2, offset=1)

    assert [item.name for item in items] == ["c", "b"]


async def test_list_past_the_end_returns_empty() -> None:
    repository = InMemoryRepository()
    await repository.add(make_item())

    assert await repository.list(offset=10) == []


async def test_count_ignores_pagination() -> None:
    repository = InMemoryRepository()
    for name in ["a", "b", "c"]:
        await repository.add(make_item(name))

    assert await repository.count() == 3


async def test_count_is_zero_when_empty() -> None:
    assert await InMemoryRepository().count() == 0


async def test_update_replaces_the_stored_item() -> None:
    repository = InMemoryRepository()
    item = make_item("before")
    await repository.add(item)
    renamed = item.model_copy(update={"name": "after"})

    assert await repository.update(item.id, renamed) == renamed
    assert (await repository.get(item.id)).name == "after"


async def test_update_returns_none_for_an_unknown_id() -> None:
    assert await InMemoryRepository().update(uuid4(), make_item()) is None


async def test_update_rekeys_when_the_identity_changes() -> None:
    """The old key must not linger behind as a duplicate."""
    repository = InMemoryRepository()
    original = make_item()
    await repository.add(original)
    moved = original.model_copy(update={"id": uuid4()})

    await repository.update(original.id, moved)

    assert await repository.get(original.id) is None
    assert await repository.get(moved.id) == moved
    assert await repository.count() == 1


async def test_delete_reports_whether_it_removed_anything() -> None:
    repository = InMemoryRepository()
    item = make_item()
    await repository.add(item)

    assert await repository.delete(item.id) is True
    assert await repository.delete(item.id) is False


async def test_ping_is_true_for_an_in_process_store() -> None:
    assert await InMemoryRepository().ping() is True


async def test_load_is_a_no_op_by_default() -> None:
    """The base-class warm-up hook must stay callable and leave the store empty."""
    repository = InMemoryRepository()

    await repository.load()

    assert await repository.count() == 0


async def test_concurrent_adds_do_not_lose_writes() -> None:
    """The internal lock is the only thing keeping this from racing."""
    repository = InMemoryRepository()

    await asyncio.gather(*(repository.add(make_item(f"item-{i}")) for i in range(50)))

    assert await repository.count() == 50
