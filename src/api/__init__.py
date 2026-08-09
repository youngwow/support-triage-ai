from fastapi import APIRouter

from src.api.routes import (
    chat,
    health,
    telegram,
)


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(telegram.router)


__all__ = [
    "api_router",
]
