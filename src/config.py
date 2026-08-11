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

    app_name: str = "support-ticket-triage"
    environment: Literal["local", "dev", "prod"] = "local"
    debug: bool = False

    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]

    log_level: str = "INFO"

    # --- external LLM -----------------------------------------------------
    gemini_api_key: str = Field(default="", repr=False)
    gemini_model: str = "gemini-3.5-flash"
    #: Per-request ceiling, milliseconds. Without it a hung provider connection
    #: would hold the triage request open indefinitely; with it the caller gets
    #: a timeout it can degrade on.
    llm_timeout_ms: int = 10_000

    # --- retrieval --------------------------------------------------------
    hf_token: str = Field(default="", repr=False)
    embedding_model_name: str = "ai-sage/Giga-Embeddings-instruct"
    embedding_device: str = "auto"
    embedding_batch_size: int = 8
    embedding_max_length: int = 4096
    knowledge_base_dir: Path = BASE_DIR / "data" / "knowledge_base"
    retrieval_top_k: int = 4
    min_retrieval_score: float = 0.5

    # --- decision thresholds ---------------------------------------------
    # Below this the classifier's answer is treated as "unknown" and a human
    # decides. Calibrate against the golden set, not by intuition.
    min_topic_confidence: float = 0.75
    # A draft below this, or one the generator marked ungrounded, is never
    # eligible for auto-send.
    min_draft_confidence: float = 0.6

    # --- async draft path -------------------------------------------------
    draft_queue_maxsize: int = 100
    draft_worker_enabled: bool = True
    warmup_on_startup: bool = True

    # --- evaluation and reporting ----------------------------------------
    golden_set_path: Path = BASE_DIR / "data" / "tickets" / "golden_set.json"
    historical_tickets_path: Path = BASE_DIR / "data" / "tickets" / "historical_tickets.json"
    # The target from the case. The PoC deliberately misses it on the LLM path
    # and reports by how much instead of hiding it — see docs/ml.md.
    hot_path_budget_ms: int = 500


@lru_cache
def get_settings() -> Settings:
    return Settings()
