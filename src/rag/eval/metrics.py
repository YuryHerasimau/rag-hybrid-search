"""Retrieval-метрики на уровне источников (детерминированные, без API).

Реализованы явно (в духе проекта — как RRF руками): для обучения важно видеть,
что именно считается, а не прятать это в чёрный ящик. Все функции работают со
списком найденных источников (rel_path в порядке ранжирования) и множеством
релевантных источников из golden-набора.

Метрики — source-level: один файл-источник учитывается один раз, даже если из
него пришло несколько чанков (нас интересует, нашёлся ли нужный документ).
"""

from __future__ import annotations

from collections.abc import Iterable


def dedup_preserve_order(sources: Iterable[str]) -> list[str]:
    """Убирает повторы источников, сохраняя порядок ранга (несколько чанков из
    одного файла схлопываются в первую позицию этого файла)."""
    seen: set[str] = set()
    out: list[str] = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0, если хотя бы один релевантный источник попал в топ-k, иначе 0.0."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/позиция первого релевантного источника (с 1); 0, если его нет."""
    for rank, src in enumerate(retrieved, start=1):
        if src in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Доля релевантных источников, попавших в топ-k."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Доля релевантного среди топ-k (по фактическому числу позиций)."""
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)
