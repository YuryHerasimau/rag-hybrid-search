"""LLM-генерация ответа (Фаза 5): grounded-ответ со структурой и abstention.

Ответ Claude возвращается Pydantic-структурой `GroundedAnswer`:
- `answer`     — текст ответа строго по контексту;
- `citations`  — какие фрагменты (источник + дословная цитата) его подтверждают;
- `abstained`  — true, если в контексте нет ответа («в базе знаний нет»).

Структурированный вывод (`structured_predict`) даёт две гарантии против
галлюцинаций: (1) модель обязана сослаться на конкретные фрагменты, (2) мы можем
сверить источники цитат с реально найденными чанками (`unsupported_citations`).
"""

from __future__ import annotations

from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore
from llama_index.llms.anthropic import Anthropic
from pydantic import BaseModel, Field

from rag.config import settings

# Текст на случай, когда ответа в контексте нет (единый, чтобы не плодить формулировки).
NO_ANSWER = "В базе знаний нет ответа на этот вопрос."


class Citation(BaseModel):
    """Ссылка на фрагмент контекста, подтверждающий ответ."""

    source: str = Field(description="путь-источник фрагмента (строка «источник:»)")
    quote: str = Field(description="короткая дословная цитата из фрагмента")


class GroundedAnswer(BaseModel):
    """Структурированный ответ с опорой на контекст."""

    answer: str = Field(description="ответ строго по контексту; иначе текст «нет ответа»")
    citations: list[Citation] = Field(
        default_factory=list, description="фрагменты, подтверждающие ответ"
    )
    abstained: bool = Field(
        default=False, description="true, если в контексте нет ответа на вопрос"
    )


GROUNDED_PROMPT = PromptTemplate(
    "Ты — ассистент по базе знаний продукта. Отвечай ТОЛЬКО на основе "
    "пронумерованных фрагментов контекста ниже.\n"
    "Для каждого утверждения укажи источник(и) в citations: source — строка после "
    "«источник:» у фрагмента, quote — короткая дословная цитата оттуда.\n"
    "Если в контексте нет ответа — верни abstained=true, answer="
    f"«{NO_ANSWER}», citations пустым. Не выдумывай факты и не ссылайся на "
    "источники вне контекста. Отвечай на русском, кратко и по делу.\n\n"
    "Контекст:\n---------------------\n{context_str}\n---------------------\n"
    "Вопрос: {query_str}\n"
)


def format_context_for_llm(nodes: list[NodeWithScore]) -> str:
    """Готовит пронумерованный контекст с источниками для grounded-промпта.

    `источник:` содержит ТОЛЬКО rel_path (стабильный идентификатор для цитаты);
    крошка — отдельной строкой `секция:`, чтобы модель не включала её в source.
    """
    blocks: list[str] = []
    for i, nws in enumerate(nodes, 1):
        meta = nws.node.metadata
        head = f"[{i}] источник: {meta.get('rel_path', '?')}"
        crumb = meta.get("breadcrumb")
        if crumb:
            head += f"\nсекция: {crumb}"
        blocks.append(f"{head}\n{nws.node.get_content().strip()}")
    return "\n\n".join(blocks)


def unsupported_citations(
    result: GroundedAnswer, nodes: list[NodeWithScore]
) -> list[str]:
    """Источники цитат, которых НЕТ среди реально найденных чанков.

    Непустой список — сигнал галлюцинации источника: модель сослалась на то,
    чего ей не давали. Сверка устойчива к тому, что модель могла дописать к
    пути секцию: достаточно, чтобы реальный rel_path входил в строку source.
    """
    retrieved = {n.node.metadata.get("rel_path") for n in nodes if n.node.metadata.get("rel_path")}
    return [c.source for c in result.citations if not any(rp in c.source for rp in retrieved)]


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


def answer_grounded(
    nodes: list[NodeWithScore], question: str, llm: Anthropic | None = None
) -> GroundedAnswer:
    """Строит grounded-ответ по уже найденным чанкам.

    Пустой контекст → сразу abstained (не тратим вызов LLM).
    """
    if not nodes:
        return GroundedAnswer(answer=NO_ANSWER, citations=[], abstained=True)
    llm = llm or build_llm()
    return llm.structured_predict(
        GroundedAnswer,
        GROUNDED_PROMPT,
        context_str=format_context_for_llm(nodes),
        query_str=question,
    )
