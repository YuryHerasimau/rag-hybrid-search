"""Инжест корпуса: загрузка Markdown-вики с сохранением структуры заголовков."""

from rag.ingest.markdown import WikiDocument, WikiSection, load_wiki

__all__ = ["WikiDocument", "WikiSection", "load_wiki"]
