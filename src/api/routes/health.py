from fastapi import (
    APIRouter, 
    Response,
    Depends, 
    status
)

from src.services import HealthService
from src.models.responses import HealthResponse
from src.dependencies import get_health_service


router = APIRouter(
    prefix="/health", 
    tags=["health"]
)


@router.get(
    "", 
    response_model=HealthResponse, 
    summary="Liveness probe"
)
async def liveness(
    service: HealthService = Depends(get_health_service)
) -> HealthResponse:
    return service.liveness()


@router.get(
    "/ready", 
    response_model=HealthResponse, 
    summary="Readiness probe"
)
async def readiness(
    response: Response,
    service: HealthService = Depends(get_health_service)
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
