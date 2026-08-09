"""
Domain entities — the "Model" of Model-Service-Repository.

These are the objects services reason about and repositories persist. They are
deliberately independent of both HTTP (no FastAPI import) and storage (no ORM,
no driver), so the same entity can serve the REST API and any other frontend.

``Item`` is a placeholder — replace it with the real aggregate.
"""

from datetime import datetime
from uuid import UUID

from pydantic import (
    BaseModel, 
    ConfigDict, 
    Field
)
from typing import Optional

from src.utils import utc_now


class Item(BaseModel):
    """
    Example aggregate root.

    Frozen: entities are replaced, never mutated in place, which keeps the
    repository the only component that decides when state changes.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def deactivate(self) -> "Item":
        """Domain behaviour lives on the entity, not in the service."""
        return self.model_copy(update={"is_active": False, "updated_at": utc_now()})
