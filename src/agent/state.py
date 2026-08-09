from typing import Optional, TypedDict

from src.agent.schemas import GroundedAnswer, RouteDecision
from src.models.domain import ChatMessage, RetrievedChunk


class AgentState(TypedDict):
    """
    Everything the graph accumulates while handling one message.
    """

    user_id: str
    question: str
    history: list[ChatMessage]
    decision: Optional[RouteDecision]
    chunks: list[RetrievedChunk]
    tool_results: dict[str, int]
    generation: Optional[GroundedAnswer]
    answer: Optional[str]
    sources: list[str]
    escalated: bool
    escalation_reason: Optional[str]


def initial_state(*, user_id: str, question: str, history: list[ChatMessage]) -> AgentState:
    return AgentState(
        user_id=user_id,
        question=question,
        history=history,
        decision=None,
        chunks=[],
        tool_results={},
        generation=None,
        answer=None,
        sources=[],
        escalated=False,
        escalation_reason=None,
    )
