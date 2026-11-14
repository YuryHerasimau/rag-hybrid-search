# Установка и запуск

> Этот документ — про первичную установку и bootstrap. Полный список команд и
> устройство пайплайна — в [how-it-works.md](how-it-works.md).

## Предпосылки

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (менеджер пакетов/окружений)
- Баланс на **Anthropic API** (console.anthropic.com) — не подписка Claude Pro.
  Подписка Pro не даёт программного доступа к моделям.

Docker **не нужен**: Qdrant используем в embedded-режиме (локальная папка).

## Шаги

1. **Зависимости.** Лёгкое ядро (Фаза 0):
   ```bash
   uv sync
   ```
   Тяжёлый ML ставим по фазам, чтобы не грузить слабый ноут раньше времени:
   ```bash
   uv sync --group phase1   # llama-index, эмбеддинги, qdrant-client (Фаза 1)
   uv sync --group eval     # ragas (Фаза 6)
   ```

2. **Переменные окружения.** Скопировать шаблон и вписать ключ API:
   ```bash
   cp .env.example .env
   # ANTHROPIC_API_KEY=sk-ant-...   (ключ из console.anthropic.com, не Pro)
   ```

3. **Данные корпуса.** Положить выгрузку вики в `data/raw/` (Markdown) и
   `data/images/` (PNG). Каталоги в `.gitignore` — корпус не коммитим.

4. **Инжест (проверка на фикстурах уже сейчас):**
   ```bash
   uv run python -m rag.ingest --path tests/fixtures/wiki --show-sections
   # на своих данных:
   uv run python -m rag.ingest --show-sections
   ```

5. **Тесты:**
   ```bash
   uv run pytest -q
   ```

## Заметки по железу (CPU-ноут без GPU)

- Первый прогон эмбеддингов медленный — результат кэшируется, повторные быстрые.
- Reranker на CPU — самая тяжёлая часть; применяется только к top-20.
- Для скорости старт на `multilingual-e5-small`; переключение на `bge-m3` —
  через конфиг, для сравнения качества.
