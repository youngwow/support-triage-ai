from fastapi import (
    APIRouter,
    Response,
    status
)

from src.models.responses import HealthResponse
from src.dependencies import HealthServiceDep


router = APIRouter(
    prefix="/health", 
    tags=["health"]
)


@router.get(
    "", 
    response_model=HealthResponse, 
    summary="Liveness probe"
)
async def liveness(service: HealthServiceDep) -> HealthResponse:
    return service.liveness()


@router.get(
    "/ready", 
    response_model=HealthResponse, 
    summary="Readiness probe"
)
async def readiness(
    response: Response,
    service: HealthServiceDep,
) -> HealthResponse:
    """
    Full check: every downstream dependency answers.

    Returns 503 when degraded so orchestrators pull the instance out of the
    load balancer instead of sending it traffic it cannot serve.
    """
    result = await service.readiness()
    if result.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
