"""LLM-генерация ответа. Фаза 1 — простой grounded-ответ на русском.

Строгую структуру ответа с обязательными цитатами и abstention добавим в
Фазе 5. Здесь — базовый промпт: отвечать только по контексту, иначе честно
сказать, что данных нет.
"""

from __future__ import annotations

from llama_index.core import PromptTemplate
from llama_index.llms.anthropic import Anthropic

from rag.config import settings

QA_PROMPT = PromptTemplate(
    "Ты — ассистент по базе знаний продукта. Отвечай ТОЛЬКО на основе контекста ниже.\n"
    "Если в контексте нет ответа — честно скажи: «В базе знаний нет ответа на этот вопрос».\n"
    "Не выдумывай факты. Отвечай на русском, кратко и по делу.\n\n"
    "Контекст:\n---------------------\n{context_str}\n---------------------\n"
    "Вопрос: {query_str}\n"
    "Ответ: "
)


def build_llm(model: str | None = None, max_tokens: int = 1024) -> Anthropic:
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "Не задан ANTHROPIC_API_KEY. Впиши ключ в .env "
            "(ключ из console.anthropic.com, не подписка Pro)."
        )
    return Anthropic(
        model=model or settings.generation_model,
        api_key=settings.anthropic_api_key,
        max_tokens=max_tokens,
    )
