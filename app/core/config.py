"""
Application configuration using pydantic-settings.
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # LLM
    google_api_key: str = ""

    # Paths
    chroma_persist_dir: str = "./data/chroma_db"
    catalog_json_path: str = "./data/shl_catalog.json"

    # Logging
    log_level: str = "INFO"

    # Agent behaviour
    max_turns: int = 8            # Hard cap per assignment spec
    max_recommendations: int = 10  # Hard cap per spec
    min_recommendations: int = 1

    # Circuit breaker
    cb_failure_threshold: int = 5
    cb_reset_timeout: float = 60.0

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
