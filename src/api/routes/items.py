from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Path,
    Query,
    status
)

from src.models.requests import (
    ItemCreateRequest,
    ItemUpdateRequest
)
from src.models.responses import (
    ItemListResponse,
    ItemResponse
)
from src.dependencies import ItemServiceDep


router = APIRouter(
    prefix="/items", 
    tags=["items"]
)

ItemId = Annotated[UUID, Path(description="Item identifier")]


@router.get(
    "", 
    response_model=ItemListResponse, 
    summary="List items"
)
async def list_items(
    service: ItemServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ItemListResponse:
    items, total = await service.list_items(limit=limit, offset=offset)
    return ItemListResponse(
        items=[ItemResponse.from_domain(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{item_id}", 
    response_model=ItemResponse,
    summary="Get one item"
)
async def get_item(
    item_id: ItemId,
    service: ItemServiceDep,
) -> ItemResponse:
    item = await service.get_item(item_id)
    return ItemResponse.from_domain(item)


@router.post(
    "",
    response_model=ItemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an item",
)
async def create_item(
    payload: ItemCreateRequest,
    service: ItemServiceDep,
) -> ItemResponse:
    item = await service.create_item(payload)
    return ItemResponse.from_domain(item)


@router.patch(
    "/{item_id}", 
    response_model=ItemResponse, 
    summary="Update an item"
)
async def update_item(
    item_id: ItemId,
    payload: ItemUpdateRequest,
    service: ItemServiceDep,
) -> ItemResponse:
    item = await service.update_item(item_id, payload)
    return ItemResponse.from_domain(item)


@router.delete(
    "/{item_id}", 
    status_code=status.HTTP_204_NO_CONTENT, 
    summary="Delete an item"
)
async def delete_item(
    item_id: ItemId,
    service: ItemServiceDep,
) -> None:
    await service.delete_item(item_id)
