import logging
from datetime import UTC, datetime


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Set up root logging once, at startup."""
    logging.basicConfig(
        level=level.upper(),
        format=_LOG_FORMAT,
        force=True,
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def utc_now() -> datetime:
    """Timezone-aware "now". Never use ``datetime.utcnow()``."""
    return datetime.now(UTC)
