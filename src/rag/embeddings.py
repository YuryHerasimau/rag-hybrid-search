"""Фабрика модели эмбеддингов.

По умолчанию — multilingual-e5-small (лёгкая, мультиязычная, работает на CPU).
Важно: e5-модели обучены с префиксами "query:" для запроса и "passage:" для
документа — без них качество заметно падает. HuggingFaceEmbedding подставляет их
через query_instruction / text_instruction.
"""

from __future__ import annotations

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag.config import settings


def build_embed_model(model_name: str | None = None) -> HuggingFaceEmbedding:
    name = model_name or settings.embedding_model
    kwargs: dict = {"model_name": name, "device": "cpu"}
    if "e5" in name.lower():
        kwargs["query_instruction"] = "query: "
        kwargs["text_instruction"] = "passage: "
    return HuggingFaceEmbedding(**kwargs)
