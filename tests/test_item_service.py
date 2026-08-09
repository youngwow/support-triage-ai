"""Unit coverage for ``src/services/item_service.py``.

The service is exercised against a real ``InMemoryRepository`` rather than a
mock: the repository is part of the contract under test, and a fake that drifts
from ``AbstractRepository`` would hide bugs instead of finding them.
"""

from uuid import uuid4

import pytest

from src.exceptions import (
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidRequestError,
)
from src.models.domain import Item
from src.models.requests import ItemCreateRequest, ItemUpdateRequest
from src.repositories.in_memory_repository import InMemoryRepository
from src.services.item_service import ItemService


@pytest.fixture
def repository() -> InMemoryRepository:
    return InMemoryRepository()


@pytest.fixture
def service(repository: InMemoryRepository) -> ItemService:
    return ItemService(repository)


async def test_create_item_persists_and_returns_the_entity(
    service: ItemService, repository: InMemoryRepository
) -> None:
    created = await service.create_item(ItemCreateRequest(name="widget", description="d"))

    assert created.name == "widget"
    assert created.is_active is True
    assert created.created_at == created.updated_at
    assert await repository.get(created.id) == created


async def test_create_item_translates_a_collision_into_already_exists(
    service: ItemService, repository: InMemoryRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repository signals a duplicate key; the service owns the HTTP meaning."""

    async def always_collides(item: Item) -> Item:
        raise KeyError(item.id)

    monkeypatch.setattr(repository, "add", always_collides)

    with pytest.raises(EntityAlreadyExistsError):
        await service.create_item(ItemCreateRequest(name="widget"))


async def test_get_item_raises_when_missing(service: ItemService) -> None:
    with pytest.raises(EntityNotFoundError):
        await service.get_item(uuid4())


async def test_list_items_returns_the_page_and_the_total(
    service: ItemService, repository: InMemoryRepository
) -> None:
    for name in ["a", "b", "c"]:
        await service.create_item(ItemCreateRequest(name=name))

    items, total = await service.list_items(limit=2, offset=0)

    assert [item.name for item in items] == ["c", "b"]
    assert total == 3


async def test_list_items_rejects_a_limit_above_the_maximum(service: ItemService) -> None:
    with pytest.raises(InvalidRequestError):
        await service.list_items(limit=ItemService.MAX_PAGE_SIZE + 1)


async def test_list_items_rejects_a_non_positive_limit(service: ItemService) -> None:
    with pytest.raises(InvalidRequestError):
        await service.list_items(limit=0)


async def test_list_items_rejects_a_negative_offset(service: ItemService) -> None:
    with pytest.raises(InvalidRequestError):
        await service.list_items(offset=-1)


async def test_update_item_applies_a_partial_change(service: ItemService) -> None:
    created = await service.create_item(ItemCreateRequest(name="before", description="kept"))

    updated = await service.update_item(created.id, ItemUpdateRequest(name="after"))

    assert updated.name == "after"
    assert updated.description == "kept"
    assert updated.id == created.id
    assert updated.created_at == created.created_at
    assert updated.updated_at >= created.updated_at


async def test_update_item_clears_the_description_on_an_explicit_null(
    service: ItemService,
) -> None:
    created = await service.create_item(ItemCreateRequest(name="w", description="gone soon"))

    updated = await service.update_item(
        created.id, ItemUpdateRequest.model_validate({"description": None})
    )

    assert updated.description is None


async def test_update_item_drops_explicit_nulls_on_non_nullable_fields(
    service: ItemService,
) -> None:
    """Regression: ``Item`` rejects a null name/is_active, which surfaced as a 500."""
    created = await service.create_item(ItemCreateRequest(name="kept"))

    updated = await service.update_item(
        created.id,
        ItemUpdateRequest.model_validate({"name": None, "is_active": None, "description": "x"}),
    )

    assert updated.name == "kept"
    assert updated.is_active is True
    assert updated.description == "x"


async def test_update_item_rejects_an_empty_payload(service: ItemService) -> None:
    created = await service.create_item(ItemCreateRequest(name="w"))

    with pytest.raises(InvalidRequestError):
        await service.update_item(created.id, ItemUpdateRequest())


async def test_update_item_rejects_a_payload_that_is_only_nulls(service: ItemService) -> None:
    created = await service.create_item(ItemCreateRequest(name="w"))

    with pytest.raises(InvalidRequestError):
        await service.update_item(created.id, ItemUpdateRequest.model_validate({"name": None}))


async def test_update_item_raises_when_missing(service: ItemService) -> None:
    with pytest.raises(EntityNotFoundError):
        await service.update_item(uuid4(), ItemUpdateRequest(name="ghost"))


async def test_delete_item_removes_it(
    service: ItemService, repository: InMemoryRepository
) -> None:
    created = await service.create_item(ItemCreateRequest(name="w"))

    await service.delete_item(created.id)

    assert await repository.get(created.id) is None


async def test_delete_item_raises_when_missing(service: ItemService) -> None:
    with pytest.raises(EntityNotFoundError):
        await service.delete_item(uuid4())
