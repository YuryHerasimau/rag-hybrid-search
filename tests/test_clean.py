"""Тесты чистки разметки Yandex Wiki (Фаза 4)."""

from rag.ingest.clean import clean_markup
from rag.ingest.markdown import parse_markdown

# Реальный блок из экспорта: таблица "ключ | значение" с цветовым тегом внутри.
_TABLE = """\
#|
||

Статус

|

**{green}(PRODUCTION)**

||
||

Метод

|

`POST /login/integration/logistician`

||
|#
"""


def test_table_two_columns_becomes_key_value():
    out = clean_markup(_TABLE)
    assert "Статус: **PRODUCTION**" in out
    assert "Метод: `POST /login/integration/logistician`" in out
    # служебные маркеры таблицы не должны протечь в текст
    assert "#|" not in out
    assert "||" not in out
    assert "|#" not in out


def test_table_three_columns_joined_by_pipe():
    raw = "#|\n||\n\nСервис\n\n|\n\nМетод\n\n|\n\nОписание\n\n||\n|#\n"
    out = clean_markup(raw)
    assert "Сервис | Метод | Описание" in out


def test_color_tag_unwrapped():
    assert clean_markup("Режим {red}(ВАЖНО) готов") == "Режим ВАЖНО готов"


def test_header_anchor_stripped():
    assert clean_markup("## Описание {#opisanie}").strip() == "## Описание"


def test_note_directive_keeps_quoted_message():
    raw = '{% note warning "На текущий момент не задается" %}\nтело\n{% endnote %}'
    out = clean_markup(raw)
    assert "На текущий момент не задается" in out
    assert "тело" in out
    assert "{%" not in out


def test_diagram_block_dropped_entirely():
    raw = "до\n{% diagram %}\nнечитаемый-исходник-схемы\n{% enddiagram %}\nпосле"
    out = clean_markup(raw)
    assert "до" in out and "после" in out
    assert "нечитаемый" not in out


def test_html_entities_decoded():
    out = clean_markup("a &lt;b&gt; c &amp; d&nbsp;e &quot;f&quot;")
    assert out == 'a <b> c & d e "f"'


def test_code_fence_table_markers_preserved():
    # `#|` внутри блока кода — не таблица, трогать нельзя
    raw = "```\n#|\nкод\n|#\n```"
    out = clean_markup(raw)
    assert "#|" in out and "|#" in out


def test_parse_markdown_applies_cleaning():
    # интеграция: parse_markdown чистит тело до разбора на секции
    doc = parse_markdown(_TABLE, "t.md")
    assert "||" not in doc.text
    assert "Статус: **PRODUCTION**" in doc.text


def test_table_single_column_kept_as_text():
    out = clean_markup("#|\n||\n\nЕдинственная ячейка\n\n||\n|#\n")
    assert "Единственная ячейка" in out
    assert "#|" not in out and "|#" not in out


def test_unclosed_table_does_not_crash_and_renders():
    # битая таблица без закрывающего |# — не падаем, ряд всё равно разворачиваем
    out = clean_markup("#|\n||\n\nСтатус\n\n|\n\nOK\n\n||\n")
    assert "Статус: OK" in out


def test_empty_table_produces_no_garbage():
    out = clean_markup("до\n\n#|\n|#\n\nпосле")
    assert "до" in out and "после" in out
    assert "#|" not in out and "|#" not in out
