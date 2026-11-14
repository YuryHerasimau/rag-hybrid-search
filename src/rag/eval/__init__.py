"""Оценка качества (Фаза 6): golden-набор, retrieval-метрики, ablation."""

from rag.eval.dataset import GoldenItem, load_golden
from rag.eval.harness import (
    GenerationReport,
    RetrievalReport,
    evaluate_generation,
    evaluate_retrieval,
)

__all__ = [
    "GenerationReport",
    "GoldenItem",
    "RetrievalReport",
    "evaluate_generation",
    "evaluate_retrieval",
    "load_golden",
]
