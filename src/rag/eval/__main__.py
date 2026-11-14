"""CLI retrieval-ablation: прогон golden-набора по режимам поиска.

    uv run python -m rag.eval --golden deliveryservice.json

Строит ретриверы dense/bm25/hybrid (+ reranker) на ТЕКУЩЕМ индексе и печатает
таблицу retrieval-метрик. Индекс должен быть собран заранее
(uv run python -m rag.index build ...). Сравнение стратегий чанкинга — отдельно:
пересобрать индекс с нужным --chunking и прогнать снова.
"""

from __future__ import annotations

import argparse

from rich.console import Console

from rag.eval.dataset import load_golden
from rag.eval.harness import evaluate_retrieval
from rag.retrieval import build_retriever

console = Console()

# (label, mode, rerank)
_CONFIGS = [
    ("dense", "dense", False),
    ("bm25", "bm25", False),
    ("hybrid", "hybrid", False),
    ("hybrid + rerank", "hybrid", True),
]


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag.eval", description="Retrieval-ablation по golden-набору")
    parser.add_argument("--golden", default="deliveryservice.json", help="файл golden-набора")
    parser.add_argument("--top-k", type=int, default=5, help="k для метрик @k")
    parser.add_argument("--candidates", type=int, default=20, help="ширина кандидатов (hybrid/rerank)")
    args = parser.parse_args()

    golden = load_golden(args.golden)
    answerable = [g for g in golden if g.answerable and g.relevant_sources]
    console.print(f"Golden: {len(golden)} вопросов, из них оцениваемых (retrieval): {len(answerable)}\n")

    reports = []
    for label, mode, rerank in _CONFIGS:
        retriever = build_retriever(mode, top_k=args.top_k, rerank=rerank, candidates=args.candidates)
        reports.append(evaluate_retrieval(golden, retriever, config=label, k=args.top_k))

    k = args.top_k
    console.print(f"| Конфигурация | hit@{k} | MRR | recall@{k} | precision@{k} |", markup=False)
    console.print("|---|---|---|---|---|", markup=False)
    for r in reports:
        console.print(r.row(), markup=False)


if __name__ == "__main__":
    main()
