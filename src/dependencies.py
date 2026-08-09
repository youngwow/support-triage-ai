from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.config import Settings, get_settings
from src.repositories.in_memory_repository import InMemoryRepository
from src.repositories.repository_interface import AbstractRepository
from src.services.health import HealthService
from src.services.item_service import ItemService


@lru_cache
def get_item_repository() -> AbstractRepository:
    return InMemoryRepository()


def get_item_service(repository: AbstractRepository = Depends(get_item_repository)) -> ItemService:
    return ItemService(repository)


def get_health_service(settings: Settings = Depends(get_settings), repository: ItemRepositoryDep) -> HealthService:
    return HealthService(settings, repository)
