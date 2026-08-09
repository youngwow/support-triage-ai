import asyncio

from src.config import get_settings
from src.dependencies import (
    get_bot,
    get_dialog_memory,
    get_dispatcher,
    get_hr_system,
    get_knowledge_base,
)
from src.utils import configure_logging, get_logger


logger = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    bot = get_bot()
    dispatcher = get_dispatcher()
    if bot is None or dispatcher is None:
        raise SystemExit("TELEGRAM_BOT_API_KEY is not set — polling cannot start")

    await get_knowledge_base().load()
    await get_hr_system().load()
    await get_dialog_memory().load()

    await bot.delete_webhook(drop_pending_updates=True)
    logger.info("Starting long polling")
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
