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

    app_name: str = "hr-it-assistant"
    environment: Literal["local", "dev", "prod"] = "local"
    debug: bool = False

    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]

    log_level: str = "INFO"

    gemini_api_key: str = Field(default="", repr=False)
    gemini_model: str = "gemini-3.5-flash"

    telegram_bot_api_key: str = Field(default="", repr=False)
    telegram_webhook_secret: str = Field(default="", repr=False)
    telegram_webhook_url: str = ""

    hf_token: str = Field(default="", repr=False)
    embedding_model_name: str = "ai-sage/Giga-Embeddings-instruct"
    embedding_device: str = "auto"
    embedding_batch_size: int = 8

    # Tests disable this: the warm-up loads the 3B embedding model.
    warmup_on_startup: bool = True

    data_dir: Path = BASE_DIR / "data"
    retrieval_top_k: int = 4
    # Calibrated on the real corpus: legit questions score >= 0.58, off-topic
    # and injection attempts <= 0.44 (Giga-Embeddings, cosine).
    min_retrieval_score: float = 0.5
    min_route_confidence: float = 0.6
    restricted_sources: list[str] = ["salary_and_grades.md", "employee_directory.md"]
    history_max_turns: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
