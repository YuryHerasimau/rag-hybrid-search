"""Чистка разметки Yandex Wiki перед чанкингом (Фаза 4).

Экспорт Yandex Wiki тащит нестандартный синтаксис, который шумит эмбеддинги и
BM25: таблицы `#|...||`, цветовые теги `{green}(...)`, блочные директивы
`{% note %}`/`{% diagram %}`, html-сущности `&nbsp;` и якоря заголовков
`{#anchor}`. Приводим всё к обычному тексту/Markdown.

Модуль — чистый Python без зависимостей от LlamaIndex: легко тестируется и
применяется на этапе парсинга (`parse_markdown`), т.е. ДО чанкинга, эмбеддингов
и BM25 — значит улучшает все режимы поиска сразу.

Таблицы разворачиваем в плоский текст (строки `ключ: значение` для двух колонок,
иначе ячейки через ` | `): для эмбеддингов и лексического поиска это надёжнее,
чем воспроизводить Markdown-таблицы с переносами внутри ячеек.
"""

from __future__ import annotations

import html
import re

_FENCE_RE = re.compile(r"^\s*(```|~~~)")

# {% diagram %} ... {% enddiagram %} — внутри обычно нечитаемый исходник схемы;
# выкидываем блок целиком (подпись, если есть, не теряем — она вне блока).
_DIAGRAM_RE = re.compile(r"\{%\s*diagram\s*%\}.*?\{%\s*enddiagram\s*%\}", re.DOTALL)

# Прочие директивы {% ... %}: {% note warning "текст" %}, {% endnote %}, {% toc %},
# {% tree %}, {% wgrid %}. Снимаем обёртку; если внутри есть сообщение в кавычках —
# сохраняем его как текст.
_DIRECTIVE_RE = re.compile(r"\{%\s*(.*?)\s*%\}", re.DOTALL)
_QUOTED_RE = re.compile(r'"([^"]*)"')

# {green}(PRODUCTION) -> PRODUCTION. Цвет произвольный ({green}, {red}, ...).
_COLOR_RE = re.compile(r"\{[a-zA-Z]+\}\(([^)]*)\)")

# Якорь в конце заголовка: `## Описание {#opisanie}` -> `## Описание`
_ANCHOR_RE = re.compile(r"\s*\{#[^}]*\}")

# 3+ пустых строки -> одна (чистка после разворота блоков)
_BLANKS_RE = re.compile(r"\n{3,}")


def _render_table_rows(block: list[str]) -> list[str]:
    """Парсит тело таблицы (строки между `#|` и `|#`) в плоский текст.

    Грамматика Yandex Wiki: строка `||` открывает/закрывает ряд (чередуясь),
    одиночная `|` разделяет ячейки, остальные непустые строки — содержимое
    ячейки. Инлайновые `|` внутри текста не трогаем (разделители всегда на
    отдельной строке).
    """
    rows: list[list[str]] = []
    row: list[str] | None = None
    cell: list[str] = []

    def flush_cell() -> None:
        if row is not None:
            row.append(" ".join(p for p in cell if p).strip())
        cell.clear()

    for raw in block:
        s = raw.strip()
        if s == "||":
            if row is None:
                row = []  # открываем ряд
            else:
                flush_cell()
                rows.append(row)  # закрываем ряд
                row = None
        elif s == "|":
            flush_cell()
        elif s:
            cell.append(s)
    if row is not None:  # незакрытый ряд (битая таблица) — не теряем
        flush_cell()
        rows.append(row)

    lines: list[str] = []
    for r in rows:
        cells = [c for c in r if c]
        if len(cells) == 2:
            lines.append(f"{cells[0]}: {cells[1]}")
        elif cells:
            lines.append(" | ".join(cells))
    return lines


def _convert_tables(text: str) -> str:
    """Разворачивает блоки `#| ... |#` в плоский текст, не трогая код в ```-фенсах."""
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if not in_fence and line.strip() == "#|":
            j = i + 1
            block: list[str] = []
            while j < len(lines) and lines[j].strip() != "|#":
                block.append(lines[j])
                j += 1
            rendered = _render_table_rows(block)
            if rendered:
                out.append("")
                out.extend(rendered)
                out.append("")
            i = j + 1  # пропускаем закрывающую |# (или конец текста)
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _unwrap_directives(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        q = _QUOTED_RE.search(m.group(1))
        return q.group(1) if q else ""

    return _DIRECTIVE_RE.sub(repl, text)


def clean_markup(text: str) -> str:
    """Очищает текст от специфичной разметки Yandex Wiki.

    Порядок важен: сначала убираем блочные директивы (диаграммы, note), затем
    инлайновые теги, затем разворачиваем таблицы. html-сущности декодируем до
    таблиц, чтобы `&lt;`/`&amp;` уже были обычными символами в ячейках.
    """
    text = _DIAGRAM_RE.sub("", text)
    text = _unwrap_directives(text)
    text = _COLOR_RE.sub(r"\1", text)
    text = _ANCHOR_RE.sub("", text)
    text = text.replace("&nbsp;", " ")
    text = html.unescape(text)  # &lt; &gt; &quot; &amp; -> < > " &
    text = _convert_tables(text)
    text = _BLANKS_RE.sub("\n\n", text)
    return text
