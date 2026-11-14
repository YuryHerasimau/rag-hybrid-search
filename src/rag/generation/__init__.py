"""Генерация ответа поверх retrieved-контекста (Claude)."""

from rag.generation.answer import QA_PROMPT, build_llm

__all__ = ["QA_PROMPT", "build_llm"]
