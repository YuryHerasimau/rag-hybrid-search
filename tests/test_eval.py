"""Тесты eval-харнесса: метрики, golden, судья, generation (всё без API)."""

from llama_index.core.schema import NodeWithScore, TextNode

from rag.eval.dataset import GOLDEN_DIR, GoldenItem, load_golden
from rag.eval.gate import check_gate
from rag.eval.harness import _representative_subset, evaluate_generation, evaluate_retrieval
from rag.eval.judge import JudgeScore, judge_faithfulness, judge_relevancy
from rag.eval.metrics import (
    dedup_preserve_order,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.generation.answer import NO_ANSWER, Citation, GroundedAnswer

REL = {"a.md", "b.md"}


def test_dedup_preserves_rank_order():
    assert dedup_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_hit_at_k():
    assert hit_at_k(["x.md", "a.md", "y.md"], REL, k=5) == 1.0
    assert hit_at_k(["x.md", "y.md"], REL, k=5) == 0.0
    # релевантный есть, но за пределами k
    assert hit_at_k(["x.md", "y.md", "a.md"], REL, k=2) == 0.0


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank(["x.md", "a.md", "b.md"], REL) == 0.5  # первый релевант на 2-й позиции
    assert reciprocal_rank(["a.md"], REL) == 1.0
    assert reciprocal_rank(["x.md", "y.md"], REL) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a.md", "b.md", "x.md"], REL, k=5) == 1.0  # оба найдены
    assert recall_at_k(["a.md", "x.md"], REL, k=5) == 0.5  # половина
    assert recall_at_k(["a.md", "b.md"], set(), k=5) == 0.0  # нет релевантных


def test_precision_at_k_uses_actual_top_size():
    assert precision_at_k(["a.md", "x.md"], REL, k=2) == 0.5
    assert precision_at_k([], REL, k=5) == 0.0
    # при коротком результате делим на фактическое число позиций, не на k
    assert precision_at_k(["a.md"], REL, k=5) == 1.0


def test_load_golden_parses_fixture():
    items = load_golden("deliveryservice.json")
    assert items and all(isinstance(i, GoldenItem) for i in items)
    # есть и отвечаемые, и неотвечаемые (для abstention)
    assert any(i.answerable for i in items)
    assert any(not i.answerable for i in items)
    # у отвечаемых есть метки источников
    assert all(i.relevant_sources for i in items if i.answerable)


def test_load_golden_resolves_bare_name_to_golden_dir():
    assert (GOLDEN_DIR / "deliveryservice.json").exists()


class _FakeRetriever:
    """Отдаёт фиксированные источники (через ноды с rel_path в metadata)."""

    def __init__(self, sources):
        self._nodes = [
            NodeWithScore(node=TextNode(text=s, metadata={"rel_path": s}), score=1.0)
            for s in sources
        ]

    def retrieve(self, _q):
        return self._nodes


class _ScoreLLM:
    """Судья: всегда возвращает заданный score."""

    def __init__(self, score):
        self._score = score

    def structured_predict(self, output_cls, prompt, **kwargs):
        return JudgeScore(score=self._score, reason="тест")


class _GenLLM:
    """Генератор: воздерживается на off-topic, иначе даёт ответ."""

    def structured_predict(self, output_cls, prompt, **kwargs):
        q = kwargs.get("query_str", "")
        if "погода" in q or "Netflix" in q:
            return GroundedAnswer(answer=NO_ANSWER, abstained=True)
        return GroundedAnswer(
            answer="ответ по делу", citations=[Citation(source="a.md", quote="x")]
        )


def test_judge_clamps_out_of_range_scores():
    assert judge_faithfulness(_ScoreLLM(1.5), "ctx", "ans") == 1.0
    assert judge_relevancy(_ScoreLLM(-0.3), "q", "ans") == 0.0
    assert judge_faithfulness(_ScoreLLM(0.5), "ctx", "ans") == 0.5


def test_evaluate_generation_scores_abstention_and_judges_answers():
    golden = [
        GoldenItem(question="как создать заказ?", relevant_sources=["a.md"], answerable=True),
        GoldenItem(question="какая погода в Париже?", relevant_sources=[], answerable=False),
    ]
    report = evaluate_generation(
        golden,
        _FakeRetriever(["a.md"]),
        gen_llm=_GenLLM(),
        judge_llm=_ScoreLLM(1.0),
        config="test",
    )
    # оба решения о воздержании верны: ответил на отвечаемый, воздержался на off-topic
    assert report.abstention_accuracy == 1.0
    assert report.n == 2
    assert report.n_judged == 1  # судился только отвеченный отвечаемый
    assert report.faithfulness == 1.0
    assert report.answer_relevancy == 1.0


def test_evaluate_generation_penalizes_hallucinated_answer_on_offtopic():
    # генератор, который НЕ воздерживается на off-topic → abstention_accuracy падает
    class _BadGen:
        def structured_predict(self, output_cls, prompt, **kwargs):
            return GroundedAnswer(answer="выдумка", abstained=False)

    golden = [GoldenItem(question="какая погода?", relevant_sources=[], answerable=False)]
    report = evaluate_generation(
        golden, _FakeRetriever(["a.md"]), gen_llm=_BadGen(), judge_llm=_ScoreLLM(1.0), config="t"
    )
    assert report.abstention_accuracy == 0.0


_BASELINE = {
    "retrieval": {"min_hit_at_k": 0.9, "min_mrr": 0.85},
    "generation": {"min_faithfulness": 0.8, "min_abstention_accuracy": 0.8},
}


def test_gate_passes_when_all_metrics_above_thresholds():
    res = check_gate(
        _BASELINE,
        retrieval={"hit_at_k": 1.0, "mrr": 0.95},
        generation={"faithfulness": 0.95, "abstention_accuracy": 0.9},
    )
    assert res.passed
    assert res.failures == []


def test_gate_fails_and_reports_metric_below_threshold():
    res = check_gate(_BASELINE, retrieval={"hit_at_k": 0.5, "mrr": 0.95})
    assert not res.passed
    assert any("retrieval.hit_at_k" in f for f in res.failures)
    # mrr выше порога — не должен попасть в нарушения
    assert not any("mrr" in f for f in res.failures)


def test_gate_flags_missing_metric():
    res = check_gate(_BASELINE, retrieval={"mrr": 0.95})  # нет hit_at_k
    assert not res.passed
    assert any("hit_at_k" in f and "не посчитана" in f for f in res.failures)


def test_gate_skips_section_when_metrics_not_provided():
    # без generation-метрик секция generation не проверяется (напр. CI без ключа)
    res = check_gate(_BASELINE, retrieval={"hit_at_k": 1.0, "mrr": 0.95}, generation=None)
    assert res.passed


def test_gate_flags_threshold_key_without_min_prefix():
    # опечатка (hit_at_k вместо min_hit_at_k) не должна молча отключать порог
    baseline = {"retrieval": {"hit_at_k": 0.9}}
    res = check_gate(baseline, retrieval={"hit_at_k": 1.0})
    assert not res.passed
    assert any("неизвестный ключ" in f for f in res.failures)


def test_representative_subset_includes_offtopic_within_limit():
    golden = [
        GoldenItem(question=f"q{i}", relevant_sources=["x.md"], answerable=True) for i in range(6)
    ]
    golden.append(GoldenItem(question="off", relevant_sources=[], answerable=False))
    sub = _representative_subset(golden, limit=3)
    assert len(sub) == 3
    assert any(not g.answerable for g in sub)  # off-topic гарантированно в выборке


def test_representative_subset_returns_all_when_no_limit():
    golden = [GoldenItem(question="q", relevant_sources=["x.md"], answerable=True)]
    assert _representative_subset(golden, None) is golden
    assert _representative_subset(golden, 99) is golden


def test_evaluate_retrieval_aggregates_over_answerable_only():
    golden = [
        GoldenItem(question="q1", relevant_sources=["a.md"], answerable=True),
        GoldenItem(question="q2", relevant_sources=[], answerable=False),  # должен игнорироваться
    ]
    # ретривер всегда возвращает a.md первым → идеальные метрики на единственном
    # оцениваемом вопросе
    report = evaluate_retrieval(golden, _FakeRetriever(["a.md", "z.md"]), config="test", k=5)
    assert report.n == 1
    assert report.hit_at_k == 1.0
    assert report.mrr == 1.0
    assert report.recall_at_k == 1.0
