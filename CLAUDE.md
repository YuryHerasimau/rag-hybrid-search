# CLAUDE.md

Рабочие заметки для Claude Code. Полное описание проекта — в [README.md](README.md)
и [docs/](docs/); здесь только то, что нужно, чтобы корректно работать в репозитории.

## О проекте (одним абзацем)

Enterprise RAG на LlamaIndex: гибридный поиск (BM25 + dense + RRF), cross-encoder
reranker, grounded-ответы Claude с цитатами и abstention, evaluation-гейт в CI.
Главная цель — **обучение RAG инкрементально**: каждый приём замеряется в числах
(ablation-таблица в README). Корпус — русскоязычная вики в Markdown + PNG-схемы.

## Окружение и команды

- Пакетный менеджер — **uv**. Всё запускать через `uv run` (venv подхватывается сам).
- ⚠️ **Грабли:** `python -m rag.*` берёт системный Python → `No module named 'rag'`.
  Всегда `uv run python -m …`. И `uv python -m …` — это НЕ запуск (у `uv python`
  своя задача). Правильно только `uv run python -m …`.
- Зависимости ставятся **по фазам** (экономим CPU-ноут без GPU):
  `uv sync --group phase1 --group phase2` (+ `--group dev` для разработки,
  `--group eval` для ragas).

Основные команды (те же, что гоняет CI — см. `.github/workflows/eval.yml`):

```bash
uv run ruff check src tests                 # линт
uv run pytest -q                            # тесты
uv run python -m rag.ingest --path tests/fixtures/wiki          # инжест
uv run python -m rag.index build --path <path> --chunking structure   # сборка индекса
uv run python -m rag.query "вопрос" --mode hybrid --rerank     # запрос
uv run python -m rag.eval --golden <file>.json [--gate] [--generation]  # eval
```

## Конвенции

- **Язык — русский**: общение, комментарии в коде, сообщения коммитов, PR, доки.
  Техтермины/названия полей API — по-английски (как в корпусе).
- **Коммиты** в стиле `type: описание` на русском (см. `git log`:
  `docs:`, `refactor(eval):`, `feat:`).
- `data/` и `storage/` — в `.gitignore` (реальная вики и индекс в репо не попадают).
  В CI eval гоняется на **синтетическом** корпусе `eval/corpus/` («Орбита»).
- Пороги CI-гейта — в `eval/baseline.json`; eval читает конфиг именно оттуда.

## Карта кода

- `src/rag/ingest/` — разбор Markdown (`markdown.py`), чистка Yandex Wiki
  (`clean.py`), caption картинок через Claude vision (`images.py`).
- `src/rag/index.py` — сборка индекса: Qdrant (dense) + `docstore.json` (BM25).
- `src/rag/retrieval.py` — dense / BM25 / RRF-гибрид / reranker.
- `src/rag/generation/answer.py` — `GroundedAnswer` (answer + citations + abstained).
- `src/rag/eval/` — метрики без API (`metrics.py`), Haiku-судья (`judge.py`),
  прогон (`harness.py`), сверка с порогами (`gate.py`).
- `src/rag/config.py` — pydantic-settings из `.env`.

## LLM

- Генерация — **Claude Sonnet**; дешёвые задачи (caption, judge) — **Claude Haiku 4.5**.
- API-ключ — отдельный баланс на console.anthropic.com (подписка Pro API не даёт).
  Ключ в `.env` (`ANTHROPIC_API_KEY`), шаблон — `.env.example`.
