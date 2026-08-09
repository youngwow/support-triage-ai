from collections.abc import Sequence
from uuid import UUID, uuid4

from src.exceptions import (
    EntityAlreadyExistsError, 
    EntityNotFoundError, 
    InvalidRequestError
)
from src.models.domain import Item
from src.models.requests import (
    ItemCreateRequest, 
    ItemUpdateRequest
)
from src.repositories.repository_interface import AbstractRepository
from src.utils import (
    get_logger, 
    utc_now
)


logger = get_logger(__name__)


class ItemService:
    MAX_PAGE_SIZE = 100


    def __init__(self, repository: AbstractRepository) -> None:
        self._repository = repository

    async def list_items(self, *, limit: int = 50, offset: int = 0) -> tuple[Sequence[Item], int]:
        """Return one page of items plus the unpaginated total."""
        if limit < 1 or limit > self.MAX_PAGE_SIZE:
            raise InvalidRequestError(f"limit must be between 1 and {self.MAX_PAGE_SIZE}")
        if offset < 0:
            raise InvalidRequestError("offset must be non-negative")

        items = await self._repository.list(limit=limit, offset=offset)
        total = await self._repository.count()
        return items, total

    async def get_item(self, item_id: UUID) -> Item:
        item = await self._repository.get(item_id)
        if item is None:
            raise EntityNotFoundError(f"Item {item_id} not found")
        return item

    async def create_item(self, payload: ItemCreateRequest) -> Item:
        now = utc_now()
        item = Item(
            id=uuid4(),
            name=payload.name,
            description=payload.description,
            created_at=now,
            updated_at=now,
        )
        try:
            created = await self._repository.add(item)
        except KeyError as exc:
            raise EntityAlreadyExistsError(f"Item {item.id} already exists") from exc
        logger.info(f"Created item {created.id}")
        return created

    async def update_item(self, item_id: UUID, payload: ItemUpdateRequest) -> Item:
        current = await self.get_item(item_id)

        changes = payload.model_dump(exclude_unset=True)

        for field in ("name", "is_active"):
            if changes.get(field) is None:
                changes.pop(field, None)

        if not changes:
            raise InvalidRequestError("Update request must contain at least one field")

        changes["updated_at"] = utc_now()

        candidate = Item.model_validate(current.model_dump() | changes)

        updated = await self._repository.update(item_id, candidate)
        if updated is None:
            raise EntityNotFoundError(f"Item {item_id} not found")
        logger.info(f"Updated item {updated.id}")
        return updated

    async def delete_item(self, item_id: UUID) -> None:
        if not await self._repository.delete(item_id):
            raise EntityNotFoundError(f"Item {item_id} not found")
        logger.info(f"Deleted item {item_id}")
