from typing import Optional
from pydantic import (
    BaseModel, 
    ConfigDict, 
    Field
)


class ItemCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)


class ItemUpdateRequest(BaseModel):
    """Partial update — only the provided fields are applied."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str]= Field(default=None, min_length=1, max_length=200)
    description: Optional[str]= Field(default=None, max_length=2000)
    is_active: Optional[bool] = None
