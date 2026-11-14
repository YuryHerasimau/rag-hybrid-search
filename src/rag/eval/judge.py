"""LLM-судья для generation-метрик (Фаза 6, Шаг 2).

Свой лёгкий судья вместо ragas: прозрачно (видно промпт и что считается) и
дёшево (одна структурная оценка на метрику, судья — Haiku). Возвращает число
0..1 через структурный вывод (как grounded-ответ).

- faithfulness  — опирается ли ОТВЕТ строго на КОНТЕКСТ (прокси галлюцинаций);
- answer_relevancy — отвечает ли ОТВЕТ на ВОПРОС по существу.
"""

from __future__ import annotations

from llama_index.core import PromptTemplate
from pydantic import BaseModel, Field


class JudgeScore(BaseModel):
    """Оценка судьи с кратким обоснованием."""

    score: float = Field(description="оценка от 0.0 до 1.0")
    reason: str = Field(description="краткое обоснование оценки")


FAITHFULNESS_PROMPT = PromptTemplate(
    "Ты — строгий оценщик. Оцени, насколько ОТВЕТ опирается ТОЛЬКО на КОНТЕКСТ.\n"
    "1.0 — каждое утверждение ответа подтверждается контекстом; 0.5 — частично; "
    "0.0 — ответ содержит факты, которых в контексте нет.\n"
    "Верни score (0..1) и reason.\n\n"
    "КОНТЕКСТ:\n---------\n{context_str}\n---------\nОТВЕТ:\n{answer}\n"
)

RELEVANCY_PROMPT = PromptTemplate(
    "Ты — строгий оценщик. Оцени, насколько ОТВЕТ отвечает на ВОПРОС по существу.\n"
    "1.0 — прямо и полно; 0.5 — частично/с водой; 0.0 — не по делу.\n"
    "Верни score (0..1) и reason.\n\n"
    "ВОПРОС: {question}\nОТВЕТ:\n{answer}\n"
)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def judge_faithfulness(judge_llm, context_str: str, answer: str) -> float:
    verdict = judge_llm.structured_predict(
        JudgeScore, FAITHFULNESS_PROMPT, context_str=context_str, answer=answer
    )
    return _clamp(verdict.score)


def judge_relevancy(judge_llm, question: str, answer: str) -> float:
    verdict = judge_llm.structured_predict(
        JudgeScore, RELEVANCY_PROMPT, question=question, answer=answer
    )
    return _clamp(verdict.score)
