"""Генерация ответа поверх retrieved-контекста (Claude): grounded + abstention."""

from rag.generation.answer import (
    GroundedAnswer,
    answer_grounded,
    build_llm,
    unsupported_citations,
)

__all__ = ["GroundedAnswer", "answer_grounded", "build_llm", "unsupported_citations"]
