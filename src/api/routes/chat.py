from fastapi import APIRouter

from src.dependencies import AssistantServiceDep
from src.models.requests import ChatRequest
from src.models.responses import ChatResponse


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask the assistant without Telegram (demo/debug)",
)
async def chat(request: ChatRequest, service: AssistantServiceDep) -> ChatResponse:
    reply = await service.handle_message(
        chat_id=request.chat_id,
        user_id=request.user_id,
        text=request.text,
    )
    return ChatResponse.from_domain(reply)
