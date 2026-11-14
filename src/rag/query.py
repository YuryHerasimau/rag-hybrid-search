"""CLI запроса к RAG.

Примеры:
    # только retrieval (без API-ключа) — посмотреть, что находится:
    uv run python -m rag.query "как создать платёж?" --retrieve-only

    # полный ответ через Claude (нужен ANTHROPIC_API_KEY в .env):
    uv run python -m rag.query "как создать платёж?"

    # гибрид + reranker (Фаза 3), только retrieval:
    uv run python -m rag.query "как создать платёж?" --mode hybrid --rerank --retrieve-only
"""

from __future__ import annotations

import argparse

from rich.console import Console

from rag.retrieval import RETRIEVAL_MODES, build_retriever, format_source

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag.query", description="Запрос к RAG (Фаза 3: +reranker)")
    parser.add_argument("question", help="вопрос на естественном языке")
    parser.add_argument("--top-k", type=int, default=5, help="сколько чанков доставать")
    parser.add_argument(
        "--mode",
        choices=RETRIEVAL_MODES,
        default="hybrid",
        help="режим поиска: dense (Фаза 1), bm25 (лексический), hybrid (RRF, по умолч.)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="пере-оценить кандидатов cross-encoder'ом (bge-reranker-base): топ-N кандидатов → top_k",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=20,
        help="сколько кандидатов достать до reranker (по умолч. 20)",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="только показать найденные чанки, без вызова LLM (не нужен ключ)",
    )
    args = parser.parse_args()

    tag = f"{args.mode}{' + rerank' if args.rerank else ''}"
    console.print(f"[dim]режим: {tag}[/dim]")
    retriever = build_retriever(
        args.mode, top_k=args.top_k, rerank=args.rerank, candidates=args.candidates
    )
    nodes = retriever.retrieve(args.question)

    console.print(f"\n[bold]Найдено чанков:[/bold] {len(nodes)}")
    for i, n in enumerate(nodes, 1):
        # markup=False: format_source содержит [...] со скорами — не разметка rich
        console.print(f"  {i}. {format_source(n)}", markup=False)

    if args.retrieve_only:
        console.print("\n[dim]--retrieve-only: LLM не вызывается.[/dim]")
        for i, n in enumerate(nodes, 1):
            snippet = n.node.get_content().strip().replace("\n", " ")[:200]
            console.print(f"\n[cyan]#{i}[/cyan] {snippet}…")
        return

    # Полный ответ через Claude — grounded, структурированный (Фаза 5)
    from rag.generation.answer import answer_grounded, unsupported_citations

    result = answer_grounded(nodes, args.question)

    if result.abstained:
        console.print(f"\n[yellow]Воздержался:[/yellow] {result.answer}", markup=False)
        return

    console.print("\n[bold green]Ответ:[/bold green]")
    console.print(result.answer, markup=False)
    console.print("\n[dim]Цитаты:[/dim]")
    for c in result.citations:
        console.print(f"  • {c.source} — «{c.quote}»", markup=False)

    # сигнал галлюцинации: источник цитаты вне найденных чанков
    bad = unsupported_citations(result, nodes)
    if bad:
        console.print(f"\n[red]⚠ цитаты на источники вне контекста:[/red] {bad}", markup=False)


if __name__ == "__main__":
    main()
