from fastapi import APIRouter

from src.api.routes import health, metrics, tickets


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(tickets.router)
api_router.include_router(metrics.router)


__all__ = [
    "api_router"
]
