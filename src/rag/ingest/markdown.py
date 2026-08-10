"""Загрузка Markdown-вики в структурированные документы.

Каждый .md-файл разбирается на секции по заголовкам (#, ##, ###...), при этом
для каждой секции сохраняется "хлебная крошка" заголовков (header_path). Это
пригодится и для метаданных при индексации, и для structure-aware чанкинга
(Фаза 4). Парсинг корректно игнорирует '#' внутри блоков кода (```).

Модуль намеренно не зависит от LlamaIndex — это чистый Python, чтобы инжест был
лёгким и тестируемым. Конвертация в llama_index.Document появится в Фазе 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass
class WikiSection:
    """Секция документа под конкретным заголовком."""

    header_path: list[str]  # ["Заголовок H1", "Подзаголовок H2", ...]
    text: str

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.header_path) if self.header_path else "(root)"


@dataclass
class WikiDocument:
    """Один .md-файл вики."""

    rel_path: str
    title: str
    frontmatter: dict
    text: str
    sections: list[WikiSection] = field(default_factory=list)

    @property
    def n_sections(self) -> int:
        return len(self.sections)


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    """Отделяет YAML-фронтматтер (--- ... ---) от тела, если он есть."""
    if not raw.startswith("---"):
        return {}, raw
    lines = raw.splitlines()
    if lines[0].strip() != "---":
        return {}, raw
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            block = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :])
            try:
                data = yaml.safe_load(block) or {}
                if not isinstance(data, dict):
                    data = {}
            except yaml.YAMLError:
                data = {}
            return data, body
    return {}, raw


def _split_sections(body: str) -> list[WikiSection]:
    """Разбивает тело на секции по заголовкам, ведя стек header_path."""
    sections: list[WikiSection] = []
    header_stack: list[tuple[int, str]] = []  # (уровень, текст)
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(WikiSection(header_path=[h for _, h in header_stack], text=text))
        buffer.clear()

    for line in body.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue

        m = None if in_fence else _HEADER_RE.match(line)
        if m:
            flush()  # закрываем предыдущую секцию перед новым заголовком
            level = len(m.group(1))
            title = m.group(2).strip()
            # схлопываем стек до заголовков строго меньшего уровня
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, title))
        else:
            buffer.append(line)

    flush()
    return sections


def parse_markdown(raw: str, rel_path: str) -> WikiDocument:
    """Парсит содержимое одного .md-файла в WikiDocument."""
    frontmatter, body = _split_frontmatter(raw)
    sections = _split_sections(body)

    # Заголовок: frontmatter.title -> первый H1 -> имя файла
    title = str(frontmatter.get("title") or "").strip()
    if not title:
        for sec in sections:
            if sec.header_path:
                title = sec.header_path[0]
                break
    if not title:
        title = Path(rel_path).stem

    return WikiDocument(
        rel_path=rel_path,
        title=title,
        frontmatter=frontmatter,
        text=body.strip(),
        sections=sections,
    )


def load_wiki(root: str | Path) -> list[WikiDocument]:
    """Рекурсивно загружает все .md-файлы из каталога в WikiDocument."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Каталог вики не найден: {root}")

    docs: list[WikiDocument] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        raw = path.read_text(encoding="utf-8")
        docs.append(parse_markdown(raw, rel))
    return docs
