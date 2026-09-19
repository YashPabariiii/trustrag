from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Infrastructure -------------------------------------------------
    DATABASE_URL: str = "postgresql+asyncpg://trustrag:trustrag@localhost:5432/trustrag"
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Auth -----------------------------------------------------------
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_ALGORITHM: str = "HS256"

    # --- Vector store ---------------------------------------------------
    # CHROMA_PERSIST_DIR is where the Chroma *server* keeps its data (mounted
    # volume). The API talks to that server over HTTP at CHROMA_HOST:CHROMA_PORT.
    CHROMA_PERSIST_DIR: str = "/chroma/data"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001

    # --- Models ---------------------------------------------------------
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    # Override to point at an OpenAI-compatible gateway or a local stub.
    # Empty means the Groq SDK's own default (https://api.groq.com).
    GROQ_BASE_URL: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RAGAS_EVAL_MODEL: str = "llama-3.1-8b-instant"

    # --- Retrieval defaults ---------------------------------------------
    DEFAULT_CHUNK_SIZE: int = 600
    DEFAULT_CHUNK_OVERLAP: int = 100
    DEFAULT_TOP_K: int = 5
    QUERY_CACHE_TTL: int = 1800

    # --- Free tier limits -----------------------------------------------
    FREE_TIER_KB_LIMIT: int = 2
    FREE_TIER_DOC_LIMIT: int = 10
    FREE_TIER_QUERY_LIMIT: int = 50

    # --- Rate limits (slowapi) ------------------------------------------
    RATE_LIMIT_FREE: str = "20/minute"
    RATE_LIMIT_PRO: str = "200/minute"

    # --- Observability --------------------------------------------------
    # The background task that refreshes the infrastructure gauges. Off in a
    # test process that only wants the app object.
    SYSTEM_METRICS_ENABLED: bool = True
    WORKER_METRICS_PORT: int = 9100

    # --- Runtime --------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # --- Demo seed ------------------------------------------------------
    SEED_DEMO_TENANT: bool = True
    DEMO_TENANT_EMAIL: str = "demo@trustrag.io"
    DEMO_TENANT_NAME: str = "Demo Tenant"

    @property
    def chroma_url(self) -> str:
        return f"http://{self.CHROMA_HOST}:{self.CHROMA_PORT}"

    @property
    def sync_database_url(self) -> str:
        """psycopg-free sync URL — Alembic runs async, this is only for tooling."""
        return self.DATABASE_URL.replace("+asyncpg", "")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
