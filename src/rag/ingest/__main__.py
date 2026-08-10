"""CLI инжеста: загрузить вики и показать сводку по документам и секциям.

Примеры:
    python -m rag.ingest                         # каталог из настроек (data/raw)
    python -m rag.ingest --path tests/fixtures/wiki
    python -m rag.ingest --path data/raw --show-sections
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rag.config import settings
from rag.ingest.markdown import load_wiki

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag.ingest", description="Инжест Markdown-вики")
    parser.add_argument(
        "--path",
        type=Path,
        default=settings.raw_dir,
        help="каталог с .md (по умолчанию — DATA_RAW_DIR из .env)",
    )
    parser.add_argument(
        "--show-sections",
        action="store_true",
        help="вывести хлебные крошки секций каждого документа",
    )
    args = parser.parse_args()

    try:
        docs = load_wiki(args.path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(
            "Положи .md-файлы вики в этот каталог или укажи путь через --path.\n"
            "Готовый пример есть в [cyan]tests/fixtures/wiki[/cyan]."
        )
        raise SystemExit(1)

    total_sections = sum(d.n_sections for d in docs)
    console.print(
        f"Загружено [bold]{len(docs)}[/bold] документов, "
        f"[bold]{total_sections}[/bold] секций из [cyan]{args.path}[/cyan]\n"
    )

    table = Table(show_lines=False)
    table.add_column("Файл", style="cyan", no_wrap=False)
    table.add_column("Заголовок")
    table.add_column("Секций", justify="right", style="green")
    table.add_column("Символов", justify="right")
    for d in docs:
        table.add_row(d.rel_path, d.title, str(d.n_sections), str(len(d.text)))
    console.print(table)

    if args.show_sections:
        for d in docs:
            console.print(f"\n[bold cyan]{d.rel_path}[/bold cyan]")
            for sec in d.sections:
                console.print(f"  • {sec.breadcrumb}  [dim]({len(sec.text)} симв.)[/dim]")


if __name__ == "__main__":
    main()
