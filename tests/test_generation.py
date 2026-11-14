"""Тесты grounded-генерации: промпт, контекст, цитаты, abstention (без API)."""

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

import rag.generation.answer as ans
from rag.generation.answer import (
    GROUNDED_PROMPT,
    NO_ANSWER,
    Citation,
    GroundedAnswer,
    answer_grounded,
    format_context_for_llm,
    unsupported_citations,
)


def _node(rel_path: str, text: str, breadcrumb: str | None = None) -> NodeWithScore:
    meta = {"rel_path": rel_path, "title": rel_path}
    if breadcrumb:
        meta["breadcrumb"] = breadcrumb
    return NodeWithScore(node=TextNode(text=text, metadata=meta), score=0.5)


class _FakeLLM:
    """Имитация LLM: не ходит в сеть, возвращает заранее заданный результат."""

    def __init__(self, result):
        self._result = result
        self.seen: dict = {}

    def structured_predict(self, output_cls, prompt, **kwargs):
        self.seen = kwargs
        return self._result


def test_build_llm_requires_api_key(monkeypatch):
    monkeypatch.setattr(ans.settings, "anthropic_api_key", "")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ans.build_llm()


def test_build_llm_passes_model_key_and_max_tokens(monkeypatch):
    # не дёргаем реальный Anthropic — подменяем конструктор, проверяем проброс
    captured = {}

    def fake_anthropic(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(ans.settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(ans, "Anthropic", fake_anthropic)

    ans.build_llm(model="claude-test", max_tokens=42)
    assert captured["api_key"] == "sk-ant-test"
    assert captured["model"] == "claude-test"
    assert captured["max_tokens"] == 42


def test_grounded_prompt_demands_grounding_and_abstention():
    rendered = GROUNDED_PROMPT.format(context_str="КОНТЕКСТ", query_str="ВОПРОС")
    assert "ТОЛЬКО на основе" in rendered
    assert "abstained=true" in rendered
    assert "citations" in rendered
    assert "КОНТЕКСТ" in rendered and "ВОПРОС" in rendered


def test_format_context_numbers_chunks_with_source_and_breadcrumb():
    nodes = [
        _node("api/payments.md", "тело платежа", breadcrumb="API > Создание"),
        _node("auth.md", "тело авторизации"),
    ]
    ctx = format_context_for_llm(nodes)
    assert "[1] источник: api/payments.md" in ctx
    assert "секция: API > Создание" in ctx
    assert "[2] источник: auth.md" in ctx
    assert "тело платежа" in ctx and "тело авторизации" in ctx


def test_answer_grounded_short_circuits_on_empty_nodes():
    # пустой контекст → abstained без вызова LLM
    result = answer_grounded([], "любой вопрос", llm=None)
    assert result.abstained is True
    assert result.answer == NO_ANSWER
    assert result.citations == []


def test_answer_grounded_passes_context_and_question_to_llm():
    nodes = [_node("api/payments.md", "платёж создаётся методом POST")]
    canned = GroundedAnswer(
        answer="Методом POST", citations=[Citation(source="api/payments.md", quote="POST")]
    )
    fake = _FakeLLM(canned)
    result = answer_grounded(nodes, "как создать платёж?", llm=fake)
    assert result is canned
    assert "как создать платёж?" == fake.seen["query_str"]
    assert "api/payments.md" in fake.seen["context_str"]


def test_unsupported_citations_flags_hallucinated_source():
    nodes = [_node("api/payments.md", "текст")]
    result = GroundedAnswer(
        answer="…",
        citations=[
            Citation(source="api/payments.md", quote="a"),  # есть среди найденных
            Citation(source="выдуманный.md", quote="b"),  # галлюцинация источника
        ],
    )
    assert unsupported_citations(result, nodes) == ["выдуманный.md"]


def test_unsupported_citations_tolerates_appended_section():
    # модель могла дописать секцию к пути — такой source всё равно подтверждён
    nodes = [_node("api/payments.md", "текст")]
    result = GroundedAnswer(
        answer="…",
        citations=[Citation(source="api/payments.md « Создание »", quote="a")],
    )
    assert unsupported_citations(result, nodes) == []


def test_unsupported_citations_no_false_support_on_nested_path():
    # выдан auth.md; цитата на НЕвыданный api-auth.md не должна «подтверждаться»
    # по подстроке (startswith, а не substring)
    nodes = [_node("auth.md", "текст")]
    result = GroundedAnswer(answer="…", citations=[Citation(source="api-auth.md", quote="a")])
    assert unsupported_citations(result, nodes) == ["api-auth.md"]
