from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.api import api_router
from src.config import Settings, get_settings
from src.dependencies import (
    get_bot,
    get_dialog_memory,
    get_hr_system,
    get_knowledge_base,
)
from src.exceptions import AppError
from src.models.responses import ErrorResponse
from src.utils import configure_logging, get_logger


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm dependencies before the first request, release them on shutdown."""
    settings: Settings = get_settings()
    logger.info(f"Starting {settings.app_name} ({settings.environment})")

    if settings.warmup_on_startup:
        await get_knowledge_base().load()
        await get_hr_system().load()
        await get_dialog_memory().load()

    bot = get_bot()
    if bot is not None and settings.telegram_webhook_url:
        webhook_url = (
            settings.telegram_webhook_url.rstrip("/")
            + settings.api_prefix
            + "/telegram/webhook"
        )
        await bot.set_webhook(
            webhook_url,
            secret_token=settings.telegram_webhook_secret or None,
            drop_pending_updates=True,
        )
        logger.info(f"Telegram webhook registered at {webhook_url}")

    yield

    if bot is not None:
        await bot.session.close()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """App factory — tests build an isolated instance instead of importing a global."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        """Single translation point from domain errors to HTTP responses."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(code=exc.code, detail=exc.detail).model_dump(),
        )

    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()


def main() -> None:
    """Local development server: ``uv run python -m src.main``."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "local",
    )


if __name__ == "__main__":
    main()
