# Как это работает

Короткая карта проекта — чтобы не терять нить между фазами.

## Поток данных

```
.md файлы (data/raw)
   │  rag.ingest.load_wiki — парсинг Markdown, фронтматтер, header_path
   ▼
WikiDocument
   │  rag.index.to_documents + SentenceSplitter — чанкинг
   ▼
чанки (nodes)
   │  rag.embeddings — multilingual-e5-small (CPU), префиксы query:/passage:
   ▼
векторы  ──►  Qdrant embedded (storage/qdrant)   [rag.index.build_index]
   │
   │  запрос пользователя
   ▼
rag.retrieval — dense-поиск похожих чанков (топ-k)
   │
   ▼
rag.generation — Claude отвечает ТОЛЬКО по найденным чанкам (grounded-промпт)
```

CLI:
- `python -m rag.ingest --path <dir> --show-sections` — посмотреть, как парсятся страницы
- `python -m rag.index build --path <dir>` — собрать/пересобрать индекс
- `python -m rag.query "вопрос" --retrieve-only` — только поиск (без API-ключа)
- `python -m rag.query "вопрос"` — полный ответ через Claude (нужен ключ в .env)

## Артефакты на диске (всё в .gitignore)

| Путь | Что это | Можно удалять? |
|------|---------|----------------|
| `storage/qdrant/` | векторный индекс Qdrant (embedded, SQLite + файлы) | да — пересоберётся через `rag.index build` |
| `.venv/` | виртуальное окружение uv | да — восстановится через `uv sync` |
| `.env` | секреты (API-ключ) | нет — это твой ключ |
| кэш модели HF | скачанные веса e5 (вне репозитория, в кэше HuggingFace) | да — скачается заново |

> `storage.sqlite` внутри `storage/qdrant/collection/wiki/` — это внутренний файл
> Qdrant, а не твои документы. Там только векторы и метаданные (rel_path, title).

## Тестовые данные vs реальные

- `tests/fixtures/wiki/` — **синтетические примеры** (онбординг, API платежей),
  придуманы для проверки пайплайна и тестов. Это не корпус.
- `data/raw/` — сюда кладётся **реальная вики** (в git не коммитится).

## Зависимости по фазам

| Группа | Пакеты | Когда |
|--------|--------|-------|
| ядро | pydantic, pydantic-settings, python-dotenv, pyyaml, rich | Фаза 0 |
| `phase1` | llama-index-core, -embeddings-huggingface, -vector-stores-qdrant, -llms-anthropic, qdrant-client<1.19, sentence-transformers (+ torch) | Фаза 1 |
| `eval` | ragas, datasets | Фаза 6 |
| `dev` | pytest, ruff | всегда |

Установка: `uv sync` (ядро) или `uv sync --group phase1` и т.д.
