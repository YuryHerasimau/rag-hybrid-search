"""Конфигурация проекта. Значения читаются из окружения и файла .env.

.env не коммитится (см. .gitignore). Шаблон — .env.example.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Корень репозитория: src/rag/config.py -> подняться на 3 уровня
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Anthropic API (pay-as-you-go, НЕ подписка Pro)
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Модели
    generation_model: str = Field(default="claude-sonnet-5", alias="GENERATION_MODEL")
    cheap_model: str = Field(default="claude-haiku-4-5", alias="CHEAP_MODEL")
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-small", alias="EMBEDDING_MODEL"
    )
    reranker_model: str = Field(default="BAAI/bge-reranker-base", alias="RERANKER_MODEL")

    # Qdrant: используем embedded-режим (локальная папка), Docker не нужен
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="wiki", alias="QDRANT_COLLECTION")

    # Пути к данным (относительно корня репозитория)
    data_raw_dir: Path = Field(default=Path("data/raw"), alias="DATA_RAW_DIR")
    data_images_dir: Path = Field(default=Path("data/images"), alias="DATA_IMAGES_DIR")

    @property
    def raw_dir(self) -> Path:
        p = self.data_raw_dir
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def images_dir(self) -> Path:
        p = self.data_images_dir
        return p if p.is_absolute() else PROJECT_ROOT / p


settings = Settings()
