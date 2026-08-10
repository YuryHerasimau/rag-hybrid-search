"""Retriever'ы. Фаза 1 — плотный (dense) поиск по векторному индексу.

Гибрид (BM25 + RRF) и reranker подключим в Фазах 2–3; здесь базовый dense.
"""

from __future__ import annotations

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

from rag.index import load_index


def build_dense_retriever(top_k: int = 5) -> VectorIndexRetriever:
    index = load_index()
    return VectorIndexRetriever(index=index, similarity_top_k=top_k)


def format_source(node: NodeWithScore) -> str:
    meta = node.node.metadata
    loc = meta.get("rel_path", "?")
    title = meta.get("title", "")
    score = node.score if node.score is not None else 0.0
    return f"[{score:.3f}] {loc} — {title}"
