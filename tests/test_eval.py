"""Тесты eval-харнесса: retrieval-метрики и загрузка golden-набора (без API)."""

from rag.eval.dataset import GOLDEN_DIR, GoldenItem, load_golden
from rag.eval.harness import evaluate_retrieval
from rag.eval.metrics import (
    dedup_preserve_order,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

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
        from llama_index.core.schema import NodeWithScore, TextNode

        self._nodes = [
            NodeWithScore(node=TextNode(text=s, metadata={"rel_path": s}), score=1.0)
            for s in sources
        ]

    def retrieve(self, _q):
        return self._nodes


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
