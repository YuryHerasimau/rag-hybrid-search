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


def test_header_emphasis_stripped_from_breadcrumb():
    # ## **Описание** {#anchor} -> крошка «Описание» без markdown-жирности,
    # при этом тело секции сохраняет свою разметку
    raw = "# Корень\n\n## **Описание** {#opisanie}\n\nтекст **жирный** тут\n"
    doc = parse_markdown(raw, "t.md")
    crumbs = [s.breadcrumb for s in doc.sections]
    assert "Корень > Описание" in crumbs
    assert all("**" not in c for c in crumbs)
    # тело не чистим — разметка остаётся
    assert "**жирный**" in doc.text


def test_underscores_in_header_preserved():
    # подчёркивания в тех-терминах не должны съедаться
    doc = parse_markdown("# Поле payment_id\n\nтело\n", "t.md")
    assert doc.sections[0].header_path == ["Поле payment_id"]


def test_utf8_content_preserved():
    doc = next(d for d in load_wiki(FIXTURES) if d.rel_path == "api/payments.md")
    # кириллица и английские поля API читаются корректно
    assert "currency" in doc.text
    assert "минорных единицах" in doc.text


def test_malformed_frontmatter_yaml_is_ignored_not_crashes():
    # битый YAML во фронтматтере → пустой frontmatter, инжест не падает
    doc = parse_markdown("---\nfoo: [unclosed\n---\n# Заголовок\n\nтело\n", "t.md")
    assert doc.frontmatter == {}
    assert doc.title == "Заголовок"  # фолбэк на первый H1


def test_frontmatter_without_closing_treated_as_body():
    # нет второго --- → фронтматтер не выделяется, заголовок из первого H1
    doc = parse_markdown("---\ntitle: X\n# Настоящий заголовок\n\nтекст\n", "t.md")
    assert doc.frontmatter == {}
    assert doc.title == "Настоящий заголовок"


def test_load_wiki_skips_empty_documents(tmp_path):
    (tmp_path / "empty.md").write_text("   \n\n  \n", encoding="utf-8")
    (tmp_path / "real.md").write_text("# H\n\nтекст\n", encoding="utf-8")
    docs = load_wiki(tmp_path)
    assert {d.rel_path for d in docs} == {"real.md"}
