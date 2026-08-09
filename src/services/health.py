from collections.abc import Mapping
from typing import Protocol

from src import __version__
from src.config import Settings
from src.models.responses import HealthResponse
from src.utils import get_logger


logger = get_logger(__name__)


class SupportsPing(Protocol):
    """Anything the readiness probe can ask "are you alive?"."""

    async def ping(self) -> bool: ...


class HealthService:
    def __init__(
        self,
        settings: Settings,
        probes: Mapping[str, SupportsPing],
    ) -> None:
        self._settings = settings
        self._probes = probes

    def liveness(self) -> HealthResponse:
        return self._response(status="ok", checks={})

    async def readiness(self) -> HealthResponse:
        checks = {name: await self._check(name, probe) for name, probe in self._probes.items()}
        checks["gemini_config"] = "ok" if self._settings.gemini_api_key else "unconfigured"
        checks["telegram_config"] = "ok" if self._settings.telegram_bot_api_key else "unconfigured"

        # The bot cannot answer without the LLM key; a missing Telegram token
        # only disables the webhook (the REST /chat endpoint still works).
        degraded = any(value != "ok" for name, value in checks.items() if name != "telegram_config")
        return self._response(status="degraded" if degraded else "ok", checks=checks)

    async def _check(self, name: str, probe: SupportsPing) -> str:
        try:
            return "ok" if await probe.ping() else "unavailable"
        except Exception as exc:
            logger.warning(f"{name} health check failed: {exc}")
            return "error"

    def _response(self, *, status: str, checks: dict[str, str]) -> HealthResponse:
        return HealthResponse(
            status=status,
            app=self._settings.app_name,
            version=__version__,
            environment=self._settings.environment,
            checks=checks,
        )
