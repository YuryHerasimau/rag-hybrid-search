# Каталог данных

Содержимое корпуса **не коммитится** в git (см. `.gitignore`) — здесь лежат
только инструкции.

Структура:

```
data/
├── raw/       # выгрузка вики в Markdown (.md)  — не в git
├── images/    # PNG-схемы и макеты              — не в git
└── processed/ # промежуточные артефакты инжеста — не в git
```

Положи выгрузку своей вики в `data/raw/`, а картинки — в `data/images/`, затем:

```bash
python -m rag.ingest --show-sections
```

Для проверки без реальных данных есть готовый пример в
[`tests/fixtures/wiki`](../tests/fixtures/wiki):

```bash
python -m rag.ingest --path tests/fixtures/wiki --show-sections
```
