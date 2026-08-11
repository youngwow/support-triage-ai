from collections.abc import Mapping
from typing import Protocol

from src import __version__
from src.config import Settings
from src.models.responses import HealthResponse
from src.utils import get_logger


logger = get_logger(__name__)


class SupportsPing(Protocol):
    async def ping(self) -> bool: ...


#: Dependencies without which the service cannot do its job at all. Everything
#: else is reported but does not flip readiness: the whole point of the design
#: is that triage keeps working when the LLM API or the vector index is gone,
#: so a missing Gemini key or an unloaded index must not pull the instance out
#: of the load balancer.
CRITICAL_PROBES = ("tickets", "audit")


class HealthService:
    def __init__(
        self,
        settings: Settings,
        probes: Mapping[str, SupportsPing],
    ) -> None:
        self._settings = settings
        self._probes = probes

    def liveness(self) -> HealthResponse:
        """No I/O: answers "is the process alive", nothing more."""
        return self._response(status="ok", checks={})

    async def readiness(self) -> HealthResponse:
        checks = {name: await self._check(name, probe) for name, probe in self._probes.items()}
        checks["llm"] = "ok" if self._settings.gemini_api_key else "unconfigured"

        degraded = any(
            value != "ok" for name, value in checks.items() if name in CRITICAL_PROBES
        )
        return self._response(status="degraded" if degraded else "ok", checks=checks)

    async def _check(self, name: str, probe: SupportsPing) -> str:
        try:
            return "ok" if await probe.ping() else "unavailable"
        except Exception as exc:
            logger.warning(f"Health check {name} failed: {exc}")
            return "error"

    def _response(self, *, status: str, checks: dict[str, str]) -> HealthResponse:
        return HealthResponse(
            status=status,
            app=self._settings.app_name,
            version=__version__,
            environment=self._settings.environment,
            checks=checks,
        )
