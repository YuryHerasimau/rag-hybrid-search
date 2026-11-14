"""Офлайн-тесты пайплайна (без сети и без API-ключа)."""

from pathlib import Path

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag.generation.answer import QA_PROMPT
from rag.index import build_nodes, to_documents
from rag.ingest import load_wiki
from rag.ingest.markdown import WikiDocument
from rag.retrieval import RerankedRetriever, RRFHybridRetriever

FIXTURES = Path(__file__).parent / "fixtures" / "wiki"


def test_to_documents_maps_metadata():
    wiki = [WikiDocument(rel_path="api/payments.md", title="API платежей", frontmatter={}, text="тело")]
    docs = to_documents(wiki)
    assert len(docs) == 1
    assert docs[0].metadata["rel_path"] == "api/payments.md"
    assert docs[0].metadata["title"] == "API платежей"
    assert docs[0].text == "тело"


def test_build_nodes_simple_keeps_citation_metadata():
    nodes = build_nodes(load_wiki(FIXTURES), chunking="simple")
    assert nodes
    assert all(n.metadata.get("rel_path") for n in nodes)
    # simple не добавляет хлебных крошек
    assert all("breadcrumb" not in n.metadata for n in nodes)


def test_build_nodes_structure_prefixes_breadcrumb():
    nodes = build_nodes(load_wiki(FIXTURES), chunking="structure")
    assert nodes
    # у вложенной секции крошка и в метаданных, и префиксом в тексте
    nested = [n for n in nodes if " > " in n.metadata.get("breadcrumb", "")]
    assert nested, "ожидали хотя бы одну вложенную секцию во фикстурах"
    n = nested[0]
    assert n.get_content().startswith(n.metadata["breadcrumb"])


def test_build_nodes_rejects_unknown_strategy():
    import pytest

    with pytest.raises(ValueError):
        build_nodes(load_wiki(FIXTURES), chunking="magic")


def test_qa_prompt_is_grounded():
    rendered = QA_PROMPT.format(context_str="КОНТЕКСТ", query_str="ВОПРОС")
    # промпт обязан требовать опоры на контекст и разрешать «нет ответа»
    assert "ТОЛЬКО на основе контекста" in rendered
    assert "нет ответа" in rendered
    assert "КОНТЕКСТ" in rendered and "ВОПРОС" in rendered


class _FakeRetriever:
    """Отдаёт заранее заданный ранжированный список нод (имитация ретривера)."""

    def __init__(self, nodes):
        self._nodes = nodes

    def retrieve(self, _query):
        return self._nodes


def _nws(node_id: str, score: float) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=node_id, text=node_id), score=score)


def test_rrf_merges_by_node_id_and_boosts_consensus():
    # dense ставит A первым, B вторым; bm25 ставит B первым, C вторым.
    # B встречается у обоих → RRF должен поднять его выше одиночных A и C.
    dense = _FakeRetriever([_nws("A", 0.9), _nws("B", 0.8)])
    bm25 = _FakeRetriever([_nws("B", 12.0), _nws("C", 7.0)])

    hybrid = RRFHybridRetriever([dense, bm25], top_k=3, k=60)
    result = hybrid._retrieve(QueryBundle(query_str="q"))

    ids = [n.node.node_id for n in result]
    assert ids[0] == "B"  # консенсус двух ретриверов победил
    assert set(ids) == {"A", "B", "C"}
    # score(B) = 1/(60+2) + 1/(60+1); score(A) = 1/(60+1); B строго больше A
    assert result[0].score > result[1].score


class _FakeReranker:
    """Имитация cross-encoder: переставляет ноды в обратном порядке, режет до top_n."""

    def __init__(self, top_n):
        self._top_n = top_n

    def postprocess_nodes(self, nodes, query_bundle=None):
        return list(reversed(nodes))[: self._top_n]


def test_reranked_retriever_delegates_to_base_then_reranker():
    base = _FakeRetriever([_nws("A", 0.9), _nws("B", 0.8), _nws("C", 0.7)])
    reranked = RerankedRetriever(base, _FakeReranker(top_n=2))
    result = reranked._retrieve(QueryBundle(query_str="q"))

    ids = [n.node.node_id for n in result]
    # базовый отдал A,B,C; reranker развернул → C,B,A и обрезал до 2
    assert ids == ["C", "B"]
