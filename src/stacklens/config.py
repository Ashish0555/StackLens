"""Validated StackLens runtime settings. Secrets are read only from the environment."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SearchStrategy(StrEnum):
    VECTOR = "vector"
    HYBRID = "hybrid"


class GuardrailScanner(StrEnum):
    NONE = "none"
    LLM_GUARD = "llm_guard"


class RerankStrategy(StrEnum):
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"


class ChunkStrategy(StrEnum):
    CHARACTERS = "characters"
    ARTICLES = "articles"


class SessionBackend(StrEnum):
    MEMORY = "memory"
    REDIS = "redis"


def _find_project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: SecretStr = SecretStr("")
    github_token: SecretStr = SecretStr("")
    github_api_url: str = "https://api.github.com"
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = Field(default=0, ge=0, le=2)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    chunk_strategy: ChunkStrategy = ChunkStrategy.CHARACTERS
    article_max_chars: int = Field(default=4000, gt=0)
    retrieval_k: int = Field(default=8, gt=0)
    search_strategy: SearchStrategy = SearchStrategy.HYBRID
    rerank_strategy: RerankStrategy = RerankStrategy.NONE
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = Field(default=24, gt=0, le=200)
    max_searches_per_turn: int = Field(default=3, gt=0, le=10)
    guardrails_enabled: bool = True
    guardrail_scanner: GuardrailScanner = GuardrailScanner.NONE
    injection_model: str = "katanemolabs/Arch-Guard"
    scan_corpus_for_injection: bool = False
    max_question_chars: int = Field(default=2000, gt=0)
    max_answer_tokens: int = Field(default=8000, gt=0)
    max_conversation_turns: int = Field(default=8, gt=0, le=50)
    max_conversation_chars: int = Field(default=30_000, gt=0, le=200_000)
    knowledge_domain: str = Field(
        default="engineering documentation, API contracts, architecture and operational runbooks",
        min_length=3,
    )
    data_dir: Path = PROJECT_ROOT / "data"
    log_dir: Path = PROJECT_ROOT / "logs"
    database_url: str = "postgresql+psycopg://stacklens:stacklens@localhost:5432/stacklens"
    database_connect_timeout: int = Field(default=5, gt=0, le=60)
    database_pool_size: int = Field(default=5, gt=0, le=100)
    database_max_overflow: int = Field(default=10, ge=0, le=100)
    collection_name: str = "stacklens_demo"
    embedding_dimensions: int = Field(default=1536, gt=0)
    session_backend: SessionBackend = SessionBackend.MEMORY
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = Field(default=3600, gt=0)
    rate_limit: str = "60/minute"
    rate_limit_storage: str = ""
    api_workers: int = Field(default=1, gt=0, le=32)
    api_key: SecretStr = SecretStr("")
    max_retries: int = Field(default=3, ge=0, le=10)
    request_timeout_seconds: float = Field(default=60, gt=0)
    prompt_label: str = "production"
    prompt_cache_seconds: int = Field(default=60, ge=0)
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_host: str = "https://cloud.langfuse.com"
    app_db_path: Path = PROJECT_ROOT / "runtime" / "application.sqlite3"
    upload_dir: Path = PROJECT_ROOT / "runtime" / "uploads"
    ingestion_mode: Literal["local", "kafka"] = "local"
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "stacklens.ingestion"
    kafka_group_id: str = "stacklens-workers"
    max_upload_bytes: int = Field(default=5_000_000, gt=0, le=50_000_000)
    cookie_secure: bool = True
    registration_enabled: bool = True
    validation_retries: int = Field(default=1, ge=0, le=2)
    evidence_judge_enabled: bool = True
    max_review_files: int = Field(default=100, gt=0, le=300)
    max_review_file_bytes: int = Field(default=500_000, gt=0, le=5_000_000)
    max_review_context_chars: int = Field(default=120_000, gt=0, le=500_000)

    @property
    def auth_required(self) -> bool:
        return bool(self.api_key.get_secret_value())

    @property
    def tracing_enabled(self) -> bool:
        return bool(
            self.langfuse_public_key.get_secret_value()
            and self.langfuse_secret_key.get_secret_value()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
