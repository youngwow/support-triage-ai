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


SettingsDep = Annotated[Settings, Depends(get_settings)]
ItemRepositoryDep = Annotated[AbstractRepository, Depends(get_item_repository)]


def get_item_service(repository: ItemRepositoryDep) -> ItemService:
    return ItemService(repository)


def get_health_service(settings: SettingsDep, repository: ItemRepositoryDep) -> HealthService:
    return HealthService(settings, repository)


ItemServiceDep = Annotated[ItemService, Depends(get_item_service)]
HealthServiceDep = Annotated[HealthService, Depends(get_health_service)]
