from datetime import datetime
from typing import (
    Literal, 
    Self,
    Optional
)
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.domain import Item


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    version: str
    environment: str
    checks: dict[str, str] = Field(default_factory=dict)


class ItemResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, item: Item) -> Self:
        """Single place where a domain entity becomes wire format."""
        return cls(
            id=item.id,
            name=item.name,
            description=item.description,
            is_active=item.is_active,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )


class ItemListResponse(BaseModel):
    items: list[ItemResponse]
    total: int
    limit: int
    offset: int


class ErrorResponse(BaseModel):
    code: str
    detail: str
