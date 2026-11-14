"""Оценка качества (Фаза 6): golden-набор, retrieval-метрики, ablation."""

from rag.eval.dataset import GoldenItem, load_golden
from rag.eval.harness import RetrievalReport, evaluate_retrieval

__all__ = ["GoldenItem", "RetrievalReport", "evaluate_retrieval", "load_golden"]
