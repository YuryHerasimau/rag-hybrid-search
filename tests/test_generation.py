"""Тесты сборки LLM-клиента (путь ошибки без ключа + проброс параметров)."""

import pytest

import rag.generation.answer as ans


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
