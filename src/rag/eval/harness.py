"""Прогон retrieval-ablation по golden-набору (без API).

Для каждого отвечаемого вопроса достаём источники ретривером и считаем
средние hit@k / MRR / recall@k / precision@k. Это квантифицирует вклад
dense/bm25/hybrid/reranker БЕЗ обращения к LLM — генеративные метрики
(faithfulness и т.п.) добавим отдельным шагом с Haiku-судьёй.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.eval.dataset import GoldenItem
from rag.eval.metrics import (
    dedup_preserve_order,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


@dataclass
class RetrievalReport:
    config: str
    n: int  # число оценённых (отвечаемых) вопросов
    hit_at_k: float
    mrr: float
    recall_at_k: float
    precision_at_k: float
    k: int

    def row(self) -> str:
        return (
            f"| {self.config} | {self.hit_at_k:.2f} | {self.mrr:.2f} | "
            f"{self.recall_at_k:.2f} | {self.precision_at_k:.2f} |"
        )


def _retrieved_sources(retriever, question: str) -> list[str]:
    nodes = retriever.retrieve(question)
    return dedup_preserve_order(n.node.metadata.get("rel_path", "?") for n in nodes)


def evaluate_retrieval(
    golden: list[GoldenItem], retriever, config: str = "?", k: int = 5
) -> RetrievalReport:
    """Средние retrieval-метрики по отвечаемым вопросам набора."""
    items = [g for g in golden if g.answerable and g.relevant_sources]
    hit = mrr = rec = prec = 0.0
    for g in items:
        got = _retrieved_sources(retriever, g.question)
        rel = set(g.relevant_sources)
        hit += hit_at_k(got, rel, k)
        mrr += reciprocal_rank(got, rel)
        rec += recall_at_k(got, rel, k)
        prec += precision_at_k(got, rel, k)
    n = len(items) or 1
    return RetrievalReport(
        config=config,
        n=len(items),
        hit_at_k=hit / n,
        mrr=mrr / n,
        recall_at_k=rec / n,
        precision_at_k=prec / n,
        k=k,
    )
