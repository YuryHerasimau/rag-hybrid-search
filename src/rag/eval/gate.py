"""CI-гейт: сравнение метрик с порогами baseline (Фаза 6, Шаг 3).

Порог — нижняя граница: метрика ниже → регрессия → гейт красный (exit 1).
Пороги хранятся в eval/baseline.json и версионируются вместе с кодом.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rag.config import PROJECT_ROOT

BASELINE_PATH = PROJECT_ROOT / "eval" / "baseline.json"


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


def load_baseline(path: str | Path = BASELINE_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check(metrics: dict[str, float], thresholds: dict[str, float], section: str) -> list[str]:
    """Проверяет метрики против порогов min_<имя>. Возвращает список нарушений."""
    failures: list[str] = []
    for key, floor in thresholds.items():
        if not key.startswith("min_"):
            # не молчим: опечатка (hit_at_k вместо min_hit_at_k) иначе отключила бы
            # порог незаметно, и гейт был бы зелёным при регрессии
            failures.append(f"{section}: неизвестный ключ порога {key!r} (ожидается min_<метрика>)")
            continue
        name = key[len("min_") :]
        value = metrics.get(name)
        if value is None:
            failures.append(f"{section}.{name}: метрика не посчитана")
        elif value < floor:
            failures.append(f"{section}.{name}={value:.3f} < порога {floor:.3f}")
    return failures


def check_gate(
    baseline: dict,
    retrieval: dict[str, float] | None = None,
    generation: dict[str, float] | None = None,
) -> GateResult:
    """Сверяет посчитанные метрики с порогами baseline.

    retrieval/generation — словари {имя_метрики: значение}. Секция проверяется
    только если она есть и в baseline, и среди переданных метрик.
    """
    failures: list[str] = []
    if retrieval is not None and "retrieval" in baseline:
        failures += _check(retrieval, baseline["retrieval"], "retrieval")
    if generation is not None and "generation" in baseline:
        failures += _check(generation, baseline["generation"], "generation")
    return GateResult(passed=not failures, failures=failures)
