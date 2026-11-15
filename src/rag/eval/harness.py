"""Прогон retrieval-ablation по golden-набору (без API).

Для каждого отвечаемого вопроса достаём источники ретривером и считаем
средние hit@k / MRR / recall@k / precision@k. Это квантифицирует вклад
dense/bm25/hybrid/reranker БЕЗ обращения к LLM — генеративные метрики
(faithfulness и т.п.) добавим отдельным шагом с Haiku-судьёй.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.eval.dataset import GoldenItem
from rag.eval.judge import judge_faithfulness, judge_relevancy
from rag.eval.metrics import (
    dedup_preserve_order,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.generation.answer import answer_grounded, format_context_for_llm


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
        """Строка markdown-таблицы (для ablation с заголовком, см. __main__)."""
        return (
            f"| {self.config} | {self.hit_at_k:.2f} | {self.mrr:.2f} | "
            f"{self.recall_at_k:.2f} | {self.precision_at_k:.2f} |"
        )

    def summary(self) -> str:
        """Самоописательная одна строка (для gate-вывода, парно с GenerationReport)."""
        return (
            f"[{self.config}] n={self.n} | hit@{self.k}={self.hit_at_k:.2f} "
            f"| mrr={self.mrr:.2f} | recall@{self.k}={self.recall_at_k:.2f} "
            f"| precision@{self.k}={self.precision_at_k:.2f}"
        )

    def metrics(self) -> dict[str, float]:
        return {
            "hit_at_k": self.hit_at_k,
            "mrr": self.mrr,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
        }


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


@dataclass
class GenerationReport:
    config: str
    n: int  # всего прогнано вопросов
    abstention_accuracy: float  # доля верных решений «отвечать / воздержаться»
    faithfulness: float  # среднее по отвеченным отвечаемым (судья)
    answer_relevancy: float
    n_judged: int  # сколько ответов оценил судья

    def summary(self) -> str:
        return (
            f"[{self.config}] n={self.n} | abstention_acc={self.abstention_accuracy:.2f} "
            f"| faithfulness={self.faithfulness:.2f} | answer_relevancy={self.answer_relevancy:.2f} "
            f"(оценено судьёй: {self.n_judged})"
        )

    def metrics(self) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "abstention_accuracy": self.abstention_accuracy,
        }


def _representative_subset(golden: list[GoldenItem], limit: int | None) -> list[GoldenItem]:
    """Подмножество из limit вопросов с гарантией попадания off-topic (abstention)."""
    if limit is None or limit >= len(golden):
        return golden
    off = [g for g in golden if not g.answerable]
    ans = [g for g in golden if g.answerable]
    return (off + ans)[:limit]


def evaluate_generation(
    golden: list[GoldenItem],
    retriever,
    gen_llm,
    judge_llm,
    config: str = "?",
    limit: int | None = None,
) -> GenerationReport:
    """Генерирует grounded-ответы и оценивает их (API: генерация + судья).

    - abstention_accuracy — детерминированно: совпало ли решение воздержаться с
      разметкой answerable (ключевая анти-галлюцинационная метрика);
    - faithfulness / answer_relevancy — судьёй, только по отвеченным отвечаемым
      вопросам (на воздержании оценивать нечего).

    limit ограничивает число вопросов (бюджет), но выборка ПРЕДСТАВИТЕЛЬНА:
    off-topic (их мало, но они критичны для проверки abstention) идут первыми,
    иначе простой префикс golden[:limit] их бы не захватил.
    """
    items = _representative_subset(golden, limit)
    abst_correct = 0
    faith_sum = rel_sum = 0.0
    judged = 0
    for g in items:
        nodes = retriever.retrieve(g.question)
        result = answer_grounded(nodes, g.question, llm=gen_llm)
        if result.abstained == (not g.answerable):
            abst_correct += 1
        if g.answerable and not result.abstained:
            ctx = format_context_for_llm(nodes)
            faith_sum += judge_faithfulness(judge_llm, ctx, result.answer)
            rel_sum += judge_relevancy(judge_llm, g.question, result.answer)
            judged += 1
    n = len(items)
    return GenerationReport(
        config=config,
        n=n,
        abstention_accuracy=abst_correct / n if n else 0.0,
        faithfulness=faith_sum / judged if judged else 0.0,
        answer_relevancy=rel_sum / judged if judged else 0.0,
        n_judged=judged,
    )
