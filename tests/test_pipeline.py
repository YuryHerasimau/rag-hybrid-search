"""Офлайн-тесты пайплайна (без сети и без API-ключа)."""

from rag.generation.answer import QA_PROMPT
from rag.index import to_documents
from rag.ingest.markdown import WikiDocument


def test_to_documents_maps_metadata():
    wiki = [WikiDocument(rel_path="api/payments.md", title="API платежей", frontmatter={}, text="тело")]
    docs = to_documents(wiki)
    assert len(docs) == 1
    assert docs[0].metadata["rel_path"] == "api/payments.md"
    assert docs[0].metadata["title"] == "API платежей"
    assert docs[0].text == "тело"


def test_qa_prompt_is_grounded():
    rendered = QA_PROMPT.format(context_str="КОНТЕКСТ", query_str="ВОПРОС")
    # промпт обязан требовать опоры на контекст и разрешать «нет ответа»
    assert "ТОЛЬКО на основе контекста" in rendered
    assert "нет ответа" in rendered
    assert "КОНТЕКСТ" in rendered and "ВОПРОС" in rendered
