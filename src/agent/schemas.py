from typing import Literal, Optional

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """
    What the classifier node decides about an incoming question.
    """

    route: Literal["knowledge_base", "escalate"]
    needs_vacation_balance: bool = False
    needs_user_grade: bool = False
    escalation_reason: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class GroundedAnswer(BaseModel):
    """
    What the generator node produces from retrieved context.
    """

    answer: str
    sources: list[str] = Field(default_factory=list)
    is_grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
