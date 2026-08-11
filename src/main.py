import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.api import api_router
from src.config import Settings, get_settings
from src.dependencies import build_draft_service, get_draft_queue, get_knowledge_base
from src.exceptions import AppError
from src.models.responses import ErrorResponse
from src.utils import configure_logging, get_logger


logger = get_logger(__name__)


async def _draft_worker() -> None:
    """Drain the draft queue until cancelled.

    One worker is enough for a PoC; concurrency here is a knob, not a design.
    The loop swallows per-ticket failures on purpose — one malformed ticket must
    not silently kill the consumer and leave the queue filling up behind it.
    """
    service = build_draft_service()
    logger.info("Draft worker started")
    while True:
        try:
            await service.process_next()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a worker must outlive one bad ticket
            logger.exception(f"Draft worker recovered from an error: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm dependencies before the first request, release them on shutdown."""
    settings: Settings = get_settings()
    logger.info(f"Starting {settings.app_name} ({settings.environment})")

    if settings.warmup_on_startup:
        try:
            await get_knowledge_base().load()
        except Exception as exc:  # noqa: BLE001 - retrieval is optional by design
            # The triage path does not need the index. Refusing to start would
            # trade a degraded service for no service at all.
            logger.error(f"Knowledge base warm-up failed ({exc}); drafts will degrade")
    else:
        logger.warning("WARMUP_ON_STARTUP is off — retrieval is disabled, drafts will degrade")

    worker: Optional[asyncio.Task[None]] = None
    if settings.draft_worker_enabled:
        worker = asyncio.create_task(_draft_worker(), name="draft-worker")

    yield

    if worker is not None:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker

    pending = get_draft_queue().pending()
    if pending:
        # The honest version of "the queue is in-memory": say what was lost.
        logger.warning(
            f"{pending} ticket(s) were still queued at shutdown and are now lost. "
            "The in-process queue is not durable — production replaces it with "
            "RabbitMQ (see docs/architecture.md)."
        )
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
