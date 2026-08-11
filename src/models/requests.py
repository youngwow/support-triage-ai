from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.models.domain import Channel


class TicketCreateRequest(BaseModel):
    """An incoming support request, whatever channel it arrived through.

    ``extra="forbid"`` so a misspelled field is a 422 rather than a field that
    silently does nothing.
    """

    model_config = ConfigDict(extra="forbid")

    channel: Channel
    text: str = Field(min_length=1, max_length=8000)
    #: Channel-side identifier (message id, mail id). Supplying it makes the
    #: POST idempotent: retrying a delivery returns the existing ticket instead
    #: of triaging and paying for it twice.
    external_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
