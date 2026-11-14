"""Golden-набор для оценки: схема и загрузка.

Фиксированный контракт для регрессий — коммитится в eval/golden/*.json. Метки
relevant_sources проставлены вручную по slug'ам имён файлов (они однозначно
кодируют тему страницы). Вопросы с answerable=false проверяют abstention.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from rag.config import PROJECT_ROOT

GOLDEN_DIR = PROJECT_ROOT / "eval" / "golden"


class GoldenItem(BaseModel):
    """Один размеченный вопрос golden-набора."""

    question: str
    relevant_sources: list[str] = Field(default_factory=list)
    ground_truth: str | None = None
    answerable: bool = True


def load_golden(path: str | Path) -> list[GoldenItem]:
    """Загружает golden-набор из JSON. Путь — абсолютный или относительно
    eval/golden/ (можно передать просто имя файла)."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = GOLDEN_DIR / p
    data = json.loads(p.read_text(encoding="utf-8"))
    return [GoldenItem(**item) for item in data]
