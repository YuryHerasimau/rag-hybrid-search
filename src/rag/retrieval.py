"""Retriever'ы.

- Фаза 1: dense — плотный (векторный) поиск по Qdrant.
- Фаза 2: bm25 — лексический поиск по словам; hybrid — слияние dense+bm25
  через RRF (Reciprocal Rank Fusion).

reranker подключим в Фазе 3.
"""

from __future__ import annotations

import re
from collections import defaultdict

from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from rag.index import load_index, load_nodes

# --- dense (Фаза 1) -----------------------------------------------------------


def build_dense_retriever(top_k: int = 5) -> VectorIndexRetriever:
    index = load_index()
    return VectorIndexRetriever(index=index, similarity_top_k=top_k)


# --- BM25 (Фаза 2) ------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _ru_tokenizer(text: str) -> list[str]:
    """Простой юникод-токенизатор: нижний регистр + слова.

    Дефолтный токенизатор bm25s заточен под английский; наш явно и предсказуемо
    работает с кириллицей. Стемминг русского — потенциальное улучшение на потом.
    """
    return _WORD_RE.findall(text.lower())


def build_bm25_retriever(top_k: int = 5):
    from llama_index.retrievers.bm25 import BM25Retriever

    nodes = load_nodes()
    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        tokenizer=_ru_tokenizer,
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


# --- диспетчер + форматирование ----------------------------------------------


def build_retriever(mode: str = "hybrid", top_k: int = 5) -> BaseRetriever:
    if mode == "dense":
        return build_dense_retriever(top_k=top_k)
    if mode == "bm25":
        return build_bm25_retriever(top_k=top_k)
    if mode == "hybrid":
        return build_hybrid_retriever(top_k=top_k)
    raise ValueError(f"Неизвестный режим retrieval: {mode!r} (dense|bm25|hybrid)")


def format_source(node: NodeWithScore) -> str:
    meta = node.node.metadata
    loc = meta.get("rel_path", "?")
    title = meta.get("title", "")
    score = node.score if node.score is not None else 0.0
    return f"[{score:.3f}] {loc} — {title}"
