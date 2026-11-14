"""Офлайн-тесты пайплайна (без сети и без API-ключа)."""

from pathlib import Path

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

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


def test_bm25_russian_stemmer_matches_inflections(monkeypatch):
    # Стеммер должен находить документ по иной грамматической форме слова:
    # запрос в ед.ч. «платёж» → текст «API платежей» (мн.ч.) матчится по корню.
    import rag.retrieval as retr

    nodes = build_nodes(load_wiki(FIXTURES), chunking="simple")
    monkeypatch.setattr(retr, "load_nodes", lambda: nodes)
    res = retr.build_bm25_retriever(top_k=3).retrieve("платёж")
    assert any("payments" in n.node.metadata.get("rel_path", "") for n in res)


def test_qdrant_client_is_singleton(tmp_path, monkeypatch):
    # Embedded-Qdrant лочит папку: второй QdrantClient на тот же путь падает.
    # _client() обязан возвращать один и тот же экземпляр в рамках процесса —
    # иначе dense, а затем hybrid в одном процессе конфликтуют (регрессия Фазы 4).
    import rag.index as idx

    monkeypatch.setattr(idx, "STORAGE_DIR", tmp_path / "qdrant")
    idx._close_client()  # сброс возможного ранее открытого клиента
    try:
        assert idx._client() is idx._client()
    finally:
        idx._close_client()


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


def test_format_source_plain_score():
    from rag.retrieval import format_source

    nws = NodeWithScore(
        node=TextNode(text="x", metadata={"rel_path": "a.md", "title": "Док"}), score=0.5
    )
    s = format_source(nws)
    assert "[0.500]" in s and "a.md" in s and "Док" in s


def test_format_source_shows_both_scores_after_rerank():
    from rag.retrieval import format_source

    # reranker кладёт исходный скор в retrieval_score → в выводе оба
    nws = NodeWithScore(
        node=TextNode(
            text="x",
            metadata={"rel_path": "a.md", "title": "Док", "retrieval_score": 0.9},
        ),
        score=0.42,
    )
    s = format_source(nws)
    assert "rerank 0.420" in s and "retr 0.900" in s


def test_build_retriever_rejects_unknown_mode():
    import pytest

    from rag.retrieval import build_retriever

    with pytest.raises(ValueError):
        build_retriever("magic", top_k=3)  # падает в диспетчере, индекс не нужен


def test_build_retriever_threads_candidates_into_hybrid(monkeypatch):
    # регрессия ревью: --candidates должен задавать fan-out суб-ретриверов hybrid,
    # а не молча игнорироваться (раньше был жёсткий хардкод 20).
    import rag.retrieval as retr

    seen = {}
    monkeypatch.setattr(
        retr, "build_dense_retriever", lambda top_k: seen.update(dense=top_k) or _FakeRetriever([])
    )
    monkeypatch.setattr(
        retr, "build_bm25_retriever", lambda top_k: seen.update(bm25=top_k) or _FakeRetriever([])
    )
    retr.build_retriever("hybrid", top_k=5, candidates=37)
    assert seen == {"dense": 37, "bm25": 37}
