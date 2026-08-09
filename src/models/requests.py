from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class ChatRequest(BaseModel):
    """A single user message addressed to the assistant."""

    model_config = ConfigDict(extra="forbid")

    chat_id: str = Field(min_length=1, max_length=64)
    user_id: Optional[str] = Field(default=None, max_length=64)
    text: str = Field(min_length=1, max_length=4000)
