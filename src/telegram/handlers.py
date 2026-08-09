from aiogram import Dispatcher, Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from src.services.assistant_service import AssistantService


router = Router(name="assistant")

_GREETING = (
    "Привет! Я внутренний HR/IT-ассистент.\n\n"
    "Могу подсказать по отпускам, больничным, заказу техники и командировкам, "
    "а также проверить ваш остаток дней отпуска. Просто задайте вопрос."
)


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(_GREETING)


@router.message()
async def on_message(message: Message, assistant: AssistantService) -> None:
    """``assistant`` arrives via Dispatcher workflow data, injected by name."""
    if not message.text:
        await message.answer("Пожалуйста, отправьте текстовое сообщение.")
        return
    reply = await assistant.handle_message(
        chat_id=str(message.chat.id),
        user_id=str(message.from_user.id) if message.from_user else None,
        text=message.text,
    )
    await message.answer(reply.text)


def create_dispatcher(assistant: AssistantService) -> Dispatcher:
    dispatcher = Dispatcher(assistant=assistant)
    dispatcher.include_router(router)
    return dispatcher
