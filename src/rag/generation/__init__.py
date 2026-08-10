"""Генерация ответа поверх retrieved-контекста (Claude)."""

from rag.generation.answer import QA_PROMPT, build_llm

__all__ = ["build_llm", "QA_PROMPT"]
