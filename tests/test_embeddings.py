"""Тесты фабрики эмбеддингов (чистая логика kwargs, без загрузки модели)."""

from rag.embeddings import _embed_kwargs


def test_e5_model_gets_query_passage_prefixes():
    # e5 обучены с префиксами query:/passage: — без них качество падает
    k = _embed_kwargs("intfloat/multilingual-e5-small")
    assert k["query_instruction"] == "query: "
    assert k["text_instruction"] == "passage: "
    assert k["device"] == "cpu"
    assert k["model_name"] == "intfloat/multilingual-e5-small"


def test_non_e5_model_gets_no_prefixes():
    # bge-m3 префиксы не использует — не должны подставляться
    k = _embed_kwargs("BAAI/bge-m3")
    assert "query_instruction" not in k
    assert "text_instruction" not in k
    assert k["device"] == "cpu"
