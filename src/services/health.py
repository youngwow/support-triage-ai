from src import __version__
from src.config import Settings
from src.models.responses import HealthResponse
from src.repositories.repository_interface import AbstractRepository
from src.utils import get_logger


logger = get_logger(__name__)


class HealthService:
    def __init__(
        self, 
        settings: Settings, 
        repository: AbstractRepository
    ) -> None:
        self._settings = settings
        self._repository = repository

    def liveness(self) -> HealthResponse:
        return self._response(status="ok", checks={})

    async def readiness(self) -> HealthResponse:
        checks = {"repository": await self._check_repository()}
        status = "ok" if all(value == "ok" for value in checks.values()) else "degraded"
        return self._response(status=status, checks=checks)

    async def _check_repository(self) -> str:
        try:
            return "ok" if await self._repository.ping() else "unavailable"
        except Exception as exc:
            logger.warning(f"Repository health check failed: {exc}")
            return "error"

    def _response(self, *, status: str, checks: dict[str, str]) -> HealthResponse:
        return HealthResponse(
            status=status,
            app=self._settings.app_name,
            version=__version__,
            environment=self._settings.environment,
            checks=checks,
        )
