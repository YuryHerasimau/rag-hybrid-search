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

from rag.config import settings
from rag.eval.dataset import load_golden
from rag.eval.harness import evaluate_generation, evaluate_retrieval
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
    parser.add_argument(
        "--generation",
        action="store_true",
        help="также прогнать generation-метрики (API: генерация + Haiku-судья)",
    )
    parser.add_argument(
        "--limit", type=int, default=6, help="сколько вопросов для generation-eval (бюджет)"
    )
    parser.add_argument(
        "--gen-model",
        default=settings.cheap_model,
        help="модель генерации для eval (по умолч. дешёвая Haiku; можно Sonnet)",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="CI-гейт: сверить метрики с порогами eval/baseline.json, exit 1 при регрессии",
    )
    args = parser.parse_args()

    golden = load_golden(args.golden)

    if args.gate:
        _run_gate(golden, args)
        return

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

    if args.generation:
        console.print(
            f"\n[dim]Generation-eval (API): модель {args.gen_model}, судья {settings.cheap_model}, "
            f"вопросов {args.limit}[/dim]"
        )
        # plain print с flush: rich при выводе в пайп иногда теряет финальную строку
        print(_generation_report(golden, _gate_retriever(args), args).summary(), flush=True)


# Гейт-конфиг retrieval+rerank — единый для гейта и generation-eval.
_GATE_MODE, _GATE_RERANK = "hybrid", True
_GATE_CONFIG = "hybrid + rerank"


def _gate_retriever(args):
    return build_retriever(_GATE_MODE, top_k=args.top_k, rerank=_GATE_RERANK, candidates=args.candidates)


def _generation_report(golden, retriever, args):
    """Собирает LLM-судью и прогоняет generation-метрики (общее для --generation и гейта)."""
    from rag.generation.answer import build_llm

    gen_llm = build_llm(model=args.gen_model)
    judge_llm = build_llm(model=settings.cheap_model)
    return evaluate_generation(
        golden, retriever, gen_llm, judge_llm, config=_GATE_CONFIG, limit=args.limit
    )


def _run_gate(golden, args) -> None:
    """Прогоняет гейт-конфиг (hybrid+rerank) и сверяет с порогами baseline.

    Retrieval-метрики — всегда (без API). Generation-метрики — только с
    --generation (нужен ключ). exit 1 при любой регрессии."""
    import sys

    from rag.eval.gate import check_gate, load_baseline

    baseline = load_baseline()
    retriever = _gate_retriever(args)
    rr = evaluate_retrieval(golden, retriever, config=_GATE_CONFIG, k=args.top_k)
    print("RETRIEVAL:", rr.row(), flush=True)

    gen_metrics = None
    if args.generation:
        gr = _generation_report(golden, retriever, args)
        gen_metrics = gr.metrics()
        print("GENERATION:", gr.summary(), flush=True)

    result = check_gate(baseline, retrieval=rr.metrics(), generation=gen_metrics)
    for f in result.failures:
        print("FAIL:", f, flush=True)
    print("GATE:", "PASS" if result.passed else "FAIL", flush=True)
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
