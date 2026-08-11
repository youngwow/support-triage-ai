from fastapi import APIRouter

from src.dependencies import MetricsServiceDep
from src.models.responses import MetricsResponse


router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get(
    "",
    response_model=MetricsResponse,
    summary="Operational metrics projected from the audit log",
)
async def metrics(service: MetricsServiceDep) -> MetricsResponse:
    """
    JSON, not Prometheus exposition format.

    ``triage_latency_ms`` against ``hot_path_budget_ms`` is where the PoC
    reports how far it sits from the 500 ms target it deliberately does not
    meet — see ``docs/ml.md``.
    """
    return await service.snapshot()
