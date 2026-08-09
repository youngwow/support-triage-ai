from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Every knob the app reads at runtime. Nothing else may call os.environ."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "app"
    environment: Literal["local", "dev", "prod"] = "local"
    debug: bool = False

    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]

    log_level: str = "INFO"

    gemini_api_key: str = Field(default="", repr=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
