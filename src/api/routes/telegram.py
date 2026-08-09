import secrets
from typing import Annotated, Any, Optional

from aiogram import Bot, Dispatcher
from aiogram.types import Update
from fastapi import APIRouter, BackgroundTasks, Header

from src.dependencies import (
    BotDep,
    DispatcherDep,
    SettingsDep,
    UpdateDeduplicatorDep,
)
from src.exceptions import BotNotConfiguredError, WebhookForbiddenError
from src.models.responses import WebhookAck
from src.utils import get_logger


logger = get_logger(__name__)

router = APIRouter(prefix="/telegram", tags=["telegram"])

SecretTokenHeader = Annotated[
    Optional[str], Header(alias="X-Telegram-Bot-Api-Secret-Token")
]


async def _feed_update_safely(dispatcher: Dispatcher, bot: Bot, update: Update) -> None:
    """Runs after the ACK is sent — a handler crash must not go unlogged."""
    try:
        await dispatcher.feed_update(bot, update)
    except Exception:
        logger.exception(f"Failed to process Telegram update {update.update_id}")


@router.post("/webhook", response_model=WebhookAck, summary="Telegram webhook")
async def telegram_webhook(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
    settings: SettingsDep,
    bot: BotDep,
    dispatcher: DispatcherDep,
    deduplicator: UpdateDeduplicatorDep,
    secret_token: SecretTokenHeader = None,
) -> WebhookAck:
    """Fast ACK: validate, dedupe, schedule — never run the LLM pipeline here.

    The full answer takes seconds (two Gemini calls plus a 3B-model query
    embedding); Telegram re-sends updates that are not ACKed within a couple
    of seconds, so processing happens after the response is returned.
    """
    if settings.telegram_webhook_secret and not secrets.compare_digest(
        secret_token or "", settings.telegram_webhook_secret
    ):
        raise WebhookForbiddenError()
    if bot is None or dispatcher is None:
        raise BotNotConfiguredError()

    update = Update.model_validate(payload, context={"bot": bot})
    if deduplicator.seen_before(update.update_id):
        logger.info(f"Duplicate Telegram update {update.update_id} ignored")
        return WebhookAck()

    background_tasks.add_task(_feed_update_safely, dispatcher, bot, update)
    return WebhookAck()
