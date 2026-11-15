# Enterprise RAG — Hybrid Search, Grounded Generation, CI Evaluation Gate

Система вопрос-ответа по корпоративной базе знаний (Markdown-вики + PNG-схемы).
Построена на LlamaIndex и Claude API, оптимизирована для инференса на CPU без GPU
и минимизации затрат на внешний API.

**Ключевые возможности:**

- **Hybrid Search** — лексический поиск BM25 + плотный (векторный) поиск, слияние
  через Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking** — переоценка пар «запрос↔чанк» моделью
  `bge-reranker-base` поверх широкого списка кандидатов.
- **Grounded Generation** — ответ Claude возвращается структурой Pydantic с
  обязательными цитатами на исходные чанки; при недостатке контекста —
  `abstained = true` (контролируемый отказ отвечать).
- **Multimodal Ingestion** — PNG-схемы индексируются через Claude vision
  (текстовое описание) на этапе инжеста.
- **CI Evaluation Gate** — детерминированные retrieval-метрики и LLM-судья на
  фиксированном golden-наборе; PR блокируется при деградации качества ниже порога.

**Корпус:** русскоязычная Markdown-вики продукта с англоязычными техническими
терминами (названия полей API) → требуются мультиязычные эмбеддинги.

---

## Архитектура пайплайна

Индексация (offline) и обработка запроса (online):

```
INDEXING (offline)
  .md (data/raw) ──┐
                   ├─▶ clean (Yandex Wiki markup) ─▶ build_nodes ─▶ ┬─▶ Qdrant (dense, e5-small)
  PNG .files/*  ───┘        │                        (chunking:     └─▶ docstore.json (BM25)
   └─ Claude vision caption │                  simple|structure|semantic)
                            └─ WikiDocument

QUERY (online)
  query ─▶ retrieval (dense | bm25 | hybrid·RRF) ─▶ rerank (cross-encoder)
        ─▶ generation (Claude ─▶ GroundedAnswer: answer + citations + abstained)
        ─▶ grounded answer / abstention

EVALUATION
  golden set ─▶ retrieval metrics (no API) + Haiku judge (faithfulness/relevancy)
             ─▶ CI gate (baseline.json thresholds)
```

Подробная mermaid-схема потока данных — в [docs/how-it-works.md](docs/how-it-works.md).

---

## Технологический стек

| Слой | Выбор | Роль в системе |
|---|---|---|
| Retrieval-фреймворк | **LlamaIndex** | инжест, чанкинг, retrievers, RRF, обвязка reranker |
| Генерация / структура ответа | **Claude API + Pydantic** | grounded-ответ с цитатами (`structured_predict`) |
| LLM (генерация) | **Claude Sonnet** | синтез финального ответа |
| LLM (caption, judge) | **Claude Haiku 4.5** | описание картинок и LLM-оценка (низкая стоимость токена) |
| Эмбеддинги | `intfloat/multilingual-e5-small` (→ `BAAI/bge-m3`) | мультиязычные dense-векторы; e5-small — baseline для CPU |
| Vector store | **Qdrant** (embedded, локальная папка) | хранение dense-векторов без внешнего сервиса |
| Лексический поиск | LlamaIndex `BM25Retriever` (in-memory) | BM25 поверх `docstore.json` |
| Стемминг BM25 | **PyStemmer** (`russian`) | нормализация грамматических форм русского языка |
| Fusion | `RRFHybridRetriever` (RRF, `k=60`) | слияние рангов BM25 и dense (несопоставимые шкалы скоров) |
| Reranker | `BAAI/bge-reranker-base` (CPU), top-20 → top_k | cross-encoder переоценка кандидатов |
| Чанкинг | `simple` / `structure` / `semantic` | structure-aware по заголовкам vs семантический сплиттинг |
| Мультимодальность | Claude vision → текстовое описание | индексация PNG-схем без локальных vision-моделей |
| Evaluation | детерминированные метрики + Haiku-судья | retrieval- и generation-метрики без внешних зависимостей |

---

## Требования и установка

### Prerequisites

| Компонент | Требование |
|---|---|
| Python | `>= 3.10` |
| Пакетный менеджер | [uv](https://docs.astral.sh/uv/) |
| CPU | x86-64, GPU не требуется |
| RAM (baseline) | ≥ 8 ГБ — `multilingual-e5-small` (~0.5 ГБ) + `bge-reranker-base` (~1.1 ГБ) |
| RAM (bge-m3) | ≥ 16 ГБ — `bge-m3` требует ~2.3 ГБ только под веса |
| Диск | ~5 ГБ под веса моделей (кэш HuggingFace) и индекс |

Веса моделей скачиваются при первом запуске и кэшируются в HuggingFace cache.

### Переменные окружения

Конфигурация читается из `.env` (`pydantic-settings`); шаблон — `.env.example`.
Все ключи опциональны и имеют значения по умолчанию; `ANTHROPIC_API_KEY`
обязателен только для генерации, caption и LLM-судьи.

| Переменная | Назначение | Обязательна | По умолчанию |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ключ Anthropic API (pay-as-you-go, не подписка Pro) | для генерации / caption / judge | *(пусто)* |
| `GENERATION_MODEL` | модель синтеза ответа | нет | `claude-sonnet-5` |
| `CHEAP_MODEL` | модель для caption и LLM-судьи | нет | `claude-haiku-4-5` |
| `EMBEDDING_MODEL` | модель dense-эмбеддингов | нет | `intfloat/multilingual-e5-small` |
| `RERANKER_MODEL` | cross-encoder reranker | нет | `BAAI/bge-reranker-base` |
| `QDRANT_COLLECTION` | имя коллекции Qdrant | нет | `wiki` |
| `QDRANT_URL` | адрес Qdrant (embedded-режим использует локальную папку `storage/qdrant`) | нет | `http://localhost:6333` |
| `DATA_RAW_DIR` | каталог исходных `.md` | нет | `data/raw` |
| `DATA_IMAGES_DIR` | каталог изображений | нет | `data/images` |

### Установка

Зависимости разбиты на группы и ставятся по фазам, чтобы не устанавливать
ML-стек раньше необходимости:

```bash
uv sync --group phase1 --group phase2            # retrieval + hybrid (текущая фаза)
uv sync --group phase1 --group phase2 --group dev # + pytest, ruff (разработка)
cp .env.example .env                             # вписать ANTHROPIC_API_KEY
```

Состав групп (`phase1`, `phase2`, `eval`, `dev`) — в [pyproject.toml](pyproject.toml)
и [docs/how-it-works.md](docs/how-it-works.md).

> **Запуск только через `uv run`** — подхватывает окружение проекта.
> `python -m rag.*` использует системный Python (`No module named 'rag'`);
> `uv python -m …` — не команда запуска. Корректно: `uv run python -m …`.

---

## Быстрый старт и CLI

```bash
# 1. Индексация корпуса (Qdrant + docstore)
uv run python -m rag.index build --path data/raw --chunking structure
uv run python -m rag.index build --captions               # + caption PNG через Claude vision (нужен ключ)

# 2. Поиск и генерация
uv run python -m rag.query "вопрос" --retrieve-only       # только поиск (без API)
uv run python -m rag.query "вопрос" --mode dense|bm25|hybrid --rerank
uv run python -m rag.query "вопрос"                       # grounded-ответ через Claude (нужен ключ)

# 3. Оценка качества
uv run python -m rag.eval --golden deliveryservice.json              # retrieval-ablation (без API)
uv run python -m rag.eval --golden deliveryservice.json --generation # + generation-метрики (Haiku)
uv run python -m rag.eval --golden synthetic.json --gate             # CI-гейт (exit 1 при регрессии)

# 4. Тесты и линт
uv run pytest -q
uv run ruff check src tests
```

Основные флаги CLI:

- `rag.ingest --path <dir> [--show-sections]` — парсинг `.md` без построения индекса.
- `rag.index build --path <dir> --chunking simple|structure|semantic [--captions]`.
- `rag.query "..." --mode dense|bm25|hybrid [--rerank] [--candidates N] [--retrieve-only]`.
- `rag.eval --golden <file> [--generation] [--gate] [--limit N] [--gen-model <id>] [--top-k K] [--candidates N]`.

Human-in-the-loop прогон (ручная разметка ответов + сверка с LLM-судьёй):
`rag.eval.human run|score` — см. [docs/evaluation.md](docs/evaluation.md).

---

## Бенчмарки (Ablation Study)

Изолированный вклад каждого приёма в числах. Прогон на срезе `api/deliveryservice`
(77 файлов), чанкинг `structure`, `k=5`.

### Retrieval (детерминированные метрики, без API)

| Конфигурация | hit@5 | MRR | recall@5 | precision@5 |
|---|---|---|---|---|
| dense | 0.86 | **0.77** | 0.86 | 0.56 |
| bm25 | 0.71 | 0.63 | 0.68 | 0.35 |
| hybrid (BM25 + RRF) | 0.86 | 0.72 | 0.82 | 0.45 |
| hybrid + reranker | **0.93** | **0.80** | **0.93** | **0.68** |

- Reranker улучшает все метрики и даёт лучшую конфигурацию.
- BM25 в одиночку — слабейший режим.
- Hybrid по MRR/precision незначительно уступает dense: RRF-консенсус размывает
  точность dense, когда BM25 не согласен; reranker это компенсирует.
- `precision@k` при ~1 релевантном источнике на вопрос — вспомогательный сигнал.

### Эмбеддинги: e5-small vs bge-m3

Апгрейд dense-модели на том же срезе (`bm25` от эмбеддингов не зависит, не приведён).

| Конфигурация | hit@5 | MRR | recall@5 | precision@5 |
|---|---|---|---|---|
| dense — e5-small | 0.86 | 0.77 | 0.86 | 0.56 |
| dense — **bge-m3** | **0.93** | **0.81** | **0.93** | **0.65** |
| hybrid+rerank — e5-small | 0.93 | **0.80** | 0.93 | **0.68** |
| hybrid+rerank — **bge-m3** | 0.93 | 0.76 | 0.93 | 0.67 |

- На этом срезе (метрики у потолка) bge-m3 заметно поднимает **чистый dense**, но
  полный `hybrid+rerank` не меняется — reranker уже компенсирует слабость e5-dense.
- Выигрыш bge-m3 проявляется на **трудных запросах**, где e5 промахивается: в
  human-eval на полном корпусе bge-m3 поднял в top-k нужные документы по вопросам,
  на которых e5 возвращал нерелевантные файлы (фильтры раздела «Маршруты», описание
  ETA) — см. [docs/evaluation.md](docs/evaluation.md).
- Цена: сборка индекса на bge-m3 (CPU) — часы против минут у e5-small, веса ~2.3 ГБ.
  Дефолт — e5-small (быстрая итерация); bge-m3 включается `EMBEDDING_MODEL=BAAI/bge-m3`
  как опциональный апгрейд качества retrieval.

### Generation (LLM-судья на Haiku)

Прогон: `hybrid + rerank`, chunking `structure`, генерация и судья — Haiku,
16 вопросов golden-набора.

| Метрика | Значение | Что показывает |
|---|---|---|
| faithfulness | **0.95** | ответ опирается на контекст (прокси галлюцинаций) |
| answer_relevancy | 0.83 | релевантность ответа вопросу |
| abstention_accuracy | **0.94** | корректность решения отвечать / воздержаться (15/16) |

- Из 14 отвечаемых вопросов Haiku ответил на 13, оба off-topic корректно ушли в abstention.
- Генерация повышается до Sonnet флагом `--gen-model claude-sonnet-5` (дороже, меньше abstention-ошибок).

Методология и воспроизведение — [docs/evaluation.md](docs/evaluation.md).

---

## CI/CD и контроль качества

GitHub Actions ([.github/workflows/eval.yml](.github/workflows/eval.yml)), два job'а:

1. **`lint-test`** — `ruff check` + `pytest`, без загрузки моделей.
2. **`eval-gate`** (`needs: lint-test`) — сборка индекса и eval на **синтетическом**
   корпусе `eval/corpus/` (реальная вики в репозиторий не коммитится, CI
   воспроизводим без приватных данных):
   - **Retrieval-гейт** — выполняется всегда, без API.
   - **Generation-гейт** — только при наличии секрета `ANTHROPIC_API_KEY`.

Пороги заданы в [eval/baseline.json](eval/baseline.json); `rag.eval --gate`
сверяет метрики с порогами и возвращает exit 1 при регрессии → PR блокируется.

| Группа порогов | Метрики |
|---|---|
| retrieval | `hit@k ≥ 0.90`, `MRR ≥ 0.85`, `recall@k ≥ 0.90` |
| generation | `faithfulness ≥ 0.80`, `abstention_accuracy ≥ 0.80` |

`concurrency` отменяет незавершённый прогон при новом push в ту же ветку.

---

## Статус и Roadmap

**Статус:** фазы 0–6 завершены — система функциональна end-to-end (hybrid-поиск,
reranker, grounded-генерация с цитатами и abstention, мультимодальный инжест,
CI-гейт, human-in-the-loop eval). 70 тестов, `ruff` чист.

**Roadmap (опционально):**

- [ ] **Фаза 7 — API/UI + router.** FastAPI + Streamlit; query router и
  text-to-SQL как второй источник данных.
- [ ] Расширение golden- и синтетического наборов.
- [ ] Прогон `--captions` на полном реальном корпусе.

Детальная история фаз и обоснование решений — [docs/architecture.md](docs/architecture.md).

---

## Структура репозитория

```
rag-hybrid-search/
├── data/                 # корпус (в .gitignore); data/README.md — формат входных данных
├── docs/                 # документация (how-it-works, setup, architecture, evaluation)
├── src/rag/              # исходники пакета
│   ├── ingest/           # инжест Markdown-вики
│   │   ├── __main__.py   #   CLI инжеста
│   │   ├── markdown.py   #   разбор .md: фронтматтер, секции, header_path
│   │   ├── clean.py      #   чистка разметки Yandex Wiki
│   │   └── images.py     #   caption картинок через Claude vision
│   ├── generation/       # Claude: grounded-ответ GroundedAnswer
│   │   └── answer.py
│   ├── eval/             # оценка: метрики, судья, ablation, CI-гейт
│   │   ├── metrics.py    #   hit@k/MRR/recall@k/precision@k (без API)
│   │   ├── judge.py      #   Haiku-судья: faithfulness/answer_relevancy
│   │   ├── harness.py    #   прогон golden-набора
│   │   ├── human.py      #   human-in-the-loop прогон и сверка с судьёй
│   │   ├── gate.py       #   сверка с порогами baseline
│   │   └── __main__.py   #   CLI: ablation и --gate
│   ├── config.py         # pydantic-settings (.env)
│   ├── embeddings.py     # multilingual-e5-small (CPU)
│   ├── index.py          # сборка индекса: Qdrant (dense) + docstore.json (BM25)
│   ├── retrieval.py      # dense / BM25 / RRF-гибрид / reranker
│   └── query.py          # CLI запроса
├── eval/                 # golden-наборы + синтетический корпус + пороги гейта
│   ├── corpus/           #   синтетическая вики «Орбита» (для CI, коммитится)
│   ├── golden/           #   размеченные Q&A (synthetic.json, deliveryservice.json)
│   ├── human/            #   вопросы для human-in-the-loop
│   └── baseline.json     #   пороги CI-гейта
├── .github/workflows/    # eval.yml — CI-гейт (линт, тесты, eval на синтетике)
├── tests/                # pytest + fixtures/wiki (мини-корпус для юнит-тестов)
├── storage/              # индекс Qdrant + docstore (в .gitignore)
├── .env.example          # шаблон конфигурации
├── pyproject.toml        # зависимости по фазам (uv)
└── README.md
```

## Документация

- [docs/how-it-works.md](docs/how-it-works.md) — карта пайплайна, справочник CLI, артефакты на диске.
- [docs/setup.md](docs/setup.md) — установка и запуск локально.
- [docs/architecture.md](docs/architecture.md) — архитектура и обоснование решений.
- [docs/evaluation.md](docs/evaluation.md) — методология оценки и CI-гейт.

## Лицензия

TODO: добавить файл `LICENSE`. Лицензия проекта пока не определена.
