from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "audio-quality-service"
    service_version: str = "0.1.0"
    service_port: int = 8000
    log_level: str = "info"
    max_audio_mb: int = 128
    default_boundary_margin_seconds: float = 0.25
    clip_threshold: float = 0.99
    use_optional_gpu_models: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
