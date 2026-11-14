"""Human-in-the-loop прогон: ответы пайплайна по вопросам для РУЧНОЙ оценки.

Две команды:
  run   — прогнать вопросы через retrieve + grounded-ответ, показать красиво и
          записать results.json с пустыми полями под человеческий вердикт
          (+ опц. оценку Haiku-судьи, чтобы потом сверить человека с судьёй);
  score — прочитать ЗАПОЛНЕННЫЙ человеком results.json, посчитать метрики и
          согласие человека с судьёй.

Запуск на реальном корпусе (индекс собрать заранее):
  uv run python -m rag.eval.human run --questions eval/human/questions.json
  # ... вручную проставить human_* в eval/human/results.json ...
  uv run python -m rag.eval.human score --results eval/human/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from rag.config import PROJECT_ROOT, settings

HUMAN_DIR = PROJECT_ROOT / "eval" / "human"
console = Console()


def _load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # допускаем как ["вопрос", ...], так и [{"question":..., "answerable":...}]
    return [{"question": q} if isinstance(q, str) else q for q in data]


def _run(args) -> None:
    from rag.generation.answer import answer_grounded, build_llm, unsupported_citations
    from rag.retrieval import build_retriever

    questions = _load_questions(Path(args.questions))
    retriever = build_retriever("hybrid", top_k=args.top_k, rerank=True, candidates=args.candidates)
    gen_llm = build_llm(model=args.model)
    judge_llm = build_llm(model=settings.cheap_model) if args.judge else None

    results: list[dict] = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        nodes = retriever.retrieve(q)
        ans = answer_grounded(nodes, q, llm=gen_llm)
        top_sources = [n.node.metadata.get("rel_path", "?") for n in nodes]
        bad = unsupported_citations(ans, nodes)

        console.print(f"\n[bold]#{i}. {q}[/bold]", markup=False)
        if ans.abstained:
            console.print("[yellow]  воздержался:[/yellow]", ans.answer, markup=False)
        else:
            console.print("  ответ:", ans.answer, markup=False)
            console.print("  цитаты:", [c.source for c in ans.citations], markup=False)
        if bad:
            console.print("  [red]⚠ висячие источники:[/red]", bad, markup=False)
        console.print("  [dim]top-источники:[/dim]", top_sources[: args.top_k], markup=False)

        rec = {
            "question": q,
            "answerable_expected": item.get("answerable"),
            "abstained": ans.abstained,
            "answer": ans.answer,
            "citations": [c.source for c in ans.citations],
            "unsupported_citations": bad,
            "top_sources": top_sources,
            # заполнить вручную: 1 — да, 0 — нет, null — не оценено
            "human_correct": None,  # ответ фактически верный?
            "human_grounded": None,  # опирается на источники (без выдумок)?
            "human_abstention_ok": None,  # решение отвечать/воздержаться уместно?
            "note": "",
        }
        if args.judge:
            from rag.eval.judge import judge_faithfulness, judge_relevancy
            from rag.generation.answer import format_context_for_llm

            if not ans.abstained:
                ctx = format_context_for_llm(nodes)
                rec["judge_faithfulness"] = judge_faithfulness(judge_llm, ctx, ans.answer)
                rec["judge_relevancy"] = judge_relevancy(judge_llm, q, ans.answer)
        results.append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\n[green]Записано {len(results)} ответов →[/green] {out}")
    console.print("Проставь human_correct / human_grounded / human_abstention_ok, затем: score")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score(args) -> None:
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))

    correct = [r["human_correct"] for r in results if r.get("human_correct") is not None]
    grounded = [r["human_grounded"] for r in results if r.get("human_grounded") is not None]
    abst = [r["human_abstention_ok"] for r in results if r.get("human_abstention_ok") is not None]

    console.print("[bold]Человеческая оценка[/bold]")
    console.print(f"  корректность:      {_mean(correct):.2f} (оценено {len(correct)})")
    console.print(f"  grounded:          {_mean(grounded):.2f} (оценено {len(grounded)})")
    console.print(f"  abstention уместна: {_mean(abst):.2f} (оценено {len(abst)})")

    # Согласие человека с судьёй по grounded/faithfulness (если судья был)
    pairs = [
        (r["human_grounded"], r["judge_faithfulness"])
        for r in results
        if r.get("human_grounded") is not None and r.get("judge_faithfulness") is not None
    ]
    if pairs:
        # согласие: судья ≥0.5 трактуем как «grounded=1»
        agree = _mean([1.0 if (h == 1) == (j >= 0.5) else 0.0 for h, j in pairs])
        console.print(
            f"\n[bold]Согласие человек↔Haiku-судья[/bold] по grounded: {agree:.2f} (пар {len(pairs)})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag.eval.human", description="Human-in-the-loop оценка")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="прогнать вопросы и записать шаблон для ручной оценки")
    run.add_argument("--questions", default=str(HUMAN_DIR / "questions.json"))
    run.add_argument("--out", default=str(HUMAN_DIR / "results.json"))
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--candidates", type=int, default=20)
    run.add_argument("--model", default=settings.generation_model, help="модель ответа (по умолч. Sonnet)")
    run.add_argument("--judge", action="store_true", help="добавить оценку Haiku-судьи для сверки")
    run.set_defaults(func=_run)

    score = sub.add_parser("score", help="посчитать метрики из заполненного results.json")
    score.add_argument("--results", default=str(HUMAN_DIR / "results.json"))
    score.set_defaults(func=_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
