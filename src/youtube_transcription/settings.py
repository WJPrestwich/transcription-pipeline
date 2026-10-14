from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/youtube_transcription"
    # Base URL of the REST API (used by the CLI)
    api_url: str = "http://localhost:8000"
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    whisper_model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8"
    batch_size: int = 8
    # Required for the gated pyannote diarization model (pyannote/speaker-diarization-community-1)
    hf_token: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    get_settings.cache_clear()
