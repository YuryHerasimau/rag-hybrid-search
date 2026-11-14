"""Retriever'ы.

- Фаза 1: dense — плотный (векторный) поиск по Qdrant.
- Фаза 2: bm25 — лексический поиск по словам; hybrid — слияние dense+bm25
  через RRF (Reciprocal Rank Fusion).
- Фаза 3: reranker — cross-encoder (bge-reranker-base) пере-оценивает пары
  «запрос↔чанк» и наводит финальный порядок поверх широкого списка кандидатов.
"""

from __future__ import annotations

from collections import defaultdict

from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from rag.config import settings
from rag.index import load_index, load_nodes

# --- dense (Фаза 1) -----------------------------------------------------------


def build_dense_retriever(top_k: int = 5) -> VectorIndexRetriever:
    index = load_index()
    return VectorIndexRetriever(index=index, similarity_top_k=top_k)


# --- BM25 (Фаза 2 + рус. стеммер Фазы 4) --------------------------------------


def build_bm25_retriever(top_k: int = 5):
    import Stemmer  # PyStemmer
    from llama_index.retrievers.bm25 import BM25Retriever

    nodes = load_nodes()
    # Русский стеммер + стоп-слова (bm25s знает рус.): BM25 матчит разные
    # грамматические формы — запрос «подтверждения номеров» находит текст
    # «подтверждение номера». Без стемминга точный документ проваливается из
    # топа. Английские API-термины (token, integration) стеммер не ломает.
    # Заменяет устаревший кастомный tokenizer (bm25s его deprecated).
    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("russian"),
        language="russian",
    )


# --- RRF-гибрид (Фаза 2) ------------------------------------------------------


class RRFHybridRetriever(BaseRetriever):
    """Сливает несколько retriever'ов через Reciprocal Rank Fusion.

    RRF складывает не скоры (у BM25 и косинуса несопоставимые шкалы), а РАНГИ:
        score(doc) = Σ_retrievers 1 / (k + rank)
    где rank — позиция документа в выдаче ретривера (с 1), k — сглаживающая
    константа (обычно 60): гасит вклад «хвоста» и не даёт первой позиции
    доминировать абсолютно.
    """

    def __init__(self, retrievers, top_k: int = 5, k: int = 60) -> None:
        self._retrievers = list(retrievers)
        self._top_k = top_k
        self._k = k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        scores: dict[str, float] = defaultdict(float)
        nodes: dict[str, NodeWithScore] = {}

        for retriever in self._retrievers:
            results = retriever.retrieve(query_bundle)
            for rank, nws in enumerate(results, start=1):
                nid = nws.node.node_id
                scores[nid] += 1.0 / (self._k + rank)
                nodes[nid] = nws  # сам узел одинаков; храним любой

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out: list[NodeWithScore] = []
        for nid, score in ranked[: self._top_k]:
            nws = nodes[nid]
            out.append(NodeWithScore(node=nws.node, score=score))
        return out


def build_hybrid_retriever(top_k: int = 5, candidates: int = 20) -> RRFHybridRetriever:
    """dense + BM25, каждый достаёт `candidates` кандидатов, RRF → top_k.

    Кандидатов берём с запасом (по умолчанию 20 с каждого), чтобы RRF было из
    чего сливать; финально оставляем top_k.
    """
    dense = build_dense_retriever(top_k=candidates)
    bm25 = build_bm25_retriever(top_k=candidates)
    return RRFHybridRetriever([dense, bm25], top_k=top_k)


# --- reranker (Фаза 3) --------------------------------------------------------


def build_reranker(top_n: int = 5):
    """Cross-encoder bge-reranker-base: пере-оценивает пары «запрос↔чанк».

    keep_retrieval_score=True сохраняет исходный скор ретрива в
    node.metadata — удобно видеть, как reranker переставил кандидатов.
    """
    from llama_index.core.postprocessor import SentenceTransformerRerank

    return SentenceTransformerRerank(
        model=settings.reranker_model,
        top_n=top_n,
        keep_retrieval_score=True,
    )


class RerankedRetriever(BaseRetriever):
    """Достаёт широкий список базовым ретривером и пере-сортирует reranker'ом."""

    def __init__(self, base: BaseRetriever, reranker) -> None:
        self._base = base
        self._reranker = reranker
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        candidates = self._base.retrieve(query_bundle)
        return self._reranker.postprocess_nodes(candidates, query_bundle=query_bundle)


# --- диспетчер + форматирование ----------------------------------------------


def _base_retriever(mode: str, k: int) -> BaseRetriever:
    if mode == "dense":
        return build_dense_retriever(top_k=k)
    if mode == "bm25":
        return build_bm25_retriever(top_k=k)
    if mode == "hybrid":
        return build_hybrid_retriever(top_k=k)
    raise ValueError(f"Неизвестный режим retrieval: {mode!r} (dense|bm25|hybrid)")


def build_retriever(
    mode: str = "hybrid",
    top_k: int = 5,
    rerank: bool = False,
    candidates: int = 20,
) -> BaseRetriever:
    """Собирает retriever. С rerank=True: базовый достаёт `candidates`
    кандидатов, cross-encoder оставляет top_k наиболее релевантных."""
    if not rerank:
        return _base_retriever(mode, top_k)
    base = _base_retriever(mode, candidates)
    return RerankedRetriever(base, build_reranker(top_n=top_k))


def format_source(node: NodeWithScore) -> str:
    meta = node.node.metadata
    loc = meta.get("rel_path", "?")
    title = meta.get("title", "")
    score = node.score if node.score is not None else 0.0
    # если работал reranker, keep_retrieval_score кладёт исходный скор в метаданные
    retr = meta.get("retrieval_score")
    prefix = f"[rerank {score:.3f} | retr {retr:.3f}]" if retr is not None else f"[{score:.3f}]"
    return f"{prefix} {loc} — {title}"
