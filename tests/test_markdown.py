"""Тесты парсера Markdown-вики."""

from pathlib import Path

from rag.ingest.markdown import load_wiki, parse_markdown

FIXTURES = Path(__file__).parent / "fixtures" / "wiki"


def test_load_wiki_finds_all_md_recursively():
    docs = load_wiki(FIXTURES)
    rel_paths = {d.rel_path for d in docs}
    assert rel_paths == {"onboarding.md", "api/payments.md"}


def test_frontmatter_title_wins():
    doc = next(d for d in load_wiki(FIXTURES) if d.rel_path == "onboarding.md")
    assert doc.title == "Онбординг пользователя"
    assert doc.frontmatter.get("tags") == ["product", "onboarding"]


def test_title_falls_back_to_first_h1():
    # у payments.md нет фронтматтера — заголовок берётся из первого H1
    doc = next(d for d in load_wiki(FIXTURES) if d.rel_path == "api/payments.md")
    assert doc.title == "API платежей"


def test_header_path_breadcrumb_is_nested():
    doc = next(d for d in load_wiki(FIXTURES) if d.rel_path == "api/payments.md")
    breadcrumbs = [s.breadcrumb for s in doc.sections]
    assert "API платежей > Создание платежа > Параметры запроса" in breadcrumbs


def test_hash_inside_code_fence_is_not_a_header():
    raw = "# Заголовок\n\n```python\n# это комментарий, не заголовок\nx = 1\n```\n"
    doc = parse_markdown(raw, "t.md")
    # одна секция (сам заголовок); '#' внутри ``` не порождает новую
    assert doc.n_sections == 1
    assert doc.sections[0].header_path == ["Заголовок"]


def test_utf8_content_preserved():
    doc = next(d for d in load_wiki(FIXTURES) if d.rel_path == "api/payments.md")
    # кириллица и английские поля API читаются корректно
    assert "currency" in doc.text
    assert "минорных единицах" in doc.text
