# Enterprise RAG — гибридный поиск, борьба с галлюцинациями, evaluation в CI

Пет-проект senior-уровня: система вопрос-ответа по корпоративной базе знаний,
устойчивая к галлюцинациям, с гибридным поиском, переранжировкой и
автоматической оценкой качества ответов прямо в CI/CD.

**Главная цель проекта:** обучение технологиям RAG на практике. Поэтому строим
инкрементально и после каждого этапа замеряем прирост метрик — чтобы *видеть*
вклад каждого приёма (см. [ablation-таблицу](#ablation-study)).

---

## Ключевые идеи

- **Гибридный retrieval:** BM25 + плотный (векторный) поиск, слияние через
  **RRF (Reciprocal Rank Fusion)**, поверх — **cross-encoder reranker**.
- **Анти-галлюцинации:** ответ Claude возвращается структурой (Pydantic) с
  обязательными цитатами на chunk-и; если контекст не позволяет ответить —
  `abstained = true` («в базе знаний нет ответа»).
- **Динамический чанкинг:** structure-aware по Markdown-заголовкам vs
  семантический сплиттинг — сравниваем в ablation.
- **Мультимодальность:** PNG-схемы/макеты из вики прогоняются через Claude vision
  на этапе инжеста → текстовое описание индексируется как обычный узел.
- **Evaluation-as-a-gate:** Ragas-метрики на фиксированном golden-наборе; в CI
  падаем, если качество деградирует ниже порога.

## Корпус данных

Реальная база: выгрузка вики одного продукта в **Markdown** + отдельно
**PNG-схемы/макеты** для спецификаций. Язык — в основном русский, названия полей
API и техтерминов на английском → нужны **мультиязычные эмбеддинги**.

---

## Технологический стек

Подобран под ограничение: слабый ноут (ASUS ZenBook, **без GPU**), бюджет — только
несколько долларов на Anthropic API.

| Слой | Выбор | Комментарий |
|---|---|---|
| Retrieval-фреймворк | **LlamaIndex** | инжест, чанкинг, retrievers, RRF, обвязка reranker |
| Генерация / структура ответа | **Claude (API) + Pydantic** | груглинг ответа с цитатами |
| LLM (генерация) | **Claude Sonnet** | платим только за генерацию |
| LLM (дёшево: caption, judge) | **Claude Haiku 4.5** | копейки за токен |
| Эмбеддинги | `intfloat/multilingual-e5-small` → апгрейд `BAAI/bge-m3` | оба мультиязычные; e5-small быстрый на CPU, bge-m3 качественнее |
| Vector store | **Qdrant** (embedded, локальная папка) | нативный hybrid; Docker не нужен |
| BM25 | LlamaIndex `BM25Retriever` (in-memory) | прозрачно для обучения |
| Fusion | `QueryFusionRetriever` c **RRF** | скоры BM25/cosine несопоставимы напрямую |
| Reranker | `BAAI/bge-reranker-base` на CPU, top-20→top-5 | m3-версия для CPU слишком тяжёлая |
| Чанкинг | `MarkdownNodeParser` → `SemanticSplitterNodeParser` | structure-aware vs semantic |
| Картинки | Claude vision → текстовое описание при инжесте | без локальных vision-моделей |
| Observability | **Phoenix (Arize)**, локально | видно, что именно retrieved |
| Eval | **Ragas**: faithfulness, answer_relevancy, context_precision/recall | стандарт |
| API / UI (опц.) | FastAPI + Streamlit | по желанию |

> ⚠️ **Оплата:** подписка Claude Pro **не** даёт доступа к API. Нужен отдельный
> баланс на console.anthropic.com (pay-as-you-go). Для этого проекта — единицы
> долларов: эмбеддинги и reranker локальные, judge на Haiku, eval на подмножестве.

---

## Быстрый старт

Проект на [uv](https://docs.astral.sh/uv/). **Все команды — через `uv run`** (сам
подхватывает `.venv`; venv вручную создавать/активировать не нужно).

> ⚠️ Грабли: `uv python -m …` — это не запуск (у `uv python` своя задача: управление
> установками Python); голый `python -m rag.ingest` берёт системный Python →
> `No module named 'rag'`. Правильно — `uv run python -m …`.

```bash
uv sync --group phase1 --group phase2                      # зависимости (текущая фаза)
cp .env.example .env                                       # вписать ANTHROPIC_API_KEY (не подписка Pro)
uv run python -m rag.ingest --path tests/fixtures/wiki     # инжест на фикстурах
uv run python -m rag.query "вопрос" --mode hybrid --rerank
```

Установка по шагам — [docs/setup.md](docs/setup.md); полный список команд и устройство
пайплайна — [docs/how-it-works.md](docs/how-it-works.md).

---

## План реализации и прогресс

Легенда: `[ ]` не начато · `[~]` в работе · `[x]` готово

- [x] **Фаза 0 — Скелет проекта.** Репозиторий, зависимости (uv/pyproject,
  установка по фазам), конфиг (`pydantic-settings` + `.env`), каркас пакета
  `src/rag/`, парсер MD-вики с сохранением структуры заголовков + CLI
  `python -m rag.ingest`, фикстуры и тесты (6 зелёных). Qdrant — в
  embedded-режиме (локальная папка), Docker не требуется.
- [x] **Фаза 1 — Naive RAG baseline.** Dense-эмбеддинги (multilingual-e5-small,
  CPU) + простой чанкинг (SentenceSplitter) → индекс в Qdrant embedded. Retriever
  (`python -m rag.query ... --retrieve-only`) и генерация ответа через Claude
  (`python -m rag.query ...`, grounded-промпт). Численный baseline по Ragas
  заполним в Фазе 6, когда будет golden-набор.
- [x] **Фаза 2 — Hybrid.** + BM25 (лексический, `bm25s`) + слияние с dense через
  **RRF** (реализован явно в `RRFHybridRetriever`). Ноды сохраняются в
  `storage/docstore.json` при сборке; режим выбирается флагом
  `python -m rag.query ... --mode dense|bm25|hybrid`. Численный замер дельты —
  в Фазе 6 (нужен golden-набор).
- [x] **Фаза 3 — Reranker.** + `bge-reranker-base` (cross-encoder). Базовый
  ретривер достаёт широкий список кандидатов (`--candidates`, по умолч. 20),
  cross-encoder пере-оценивает пары «запрос↔чанк» и оставляет top_k:
  `python -m rag.query ... --rerank`. Численный замер дельты — в Фазе 6.
- [x] **Фаза 4 — Чанкинг + чистка.** Чистка разметки Yandex Wiki (таблицы
  `#|...||`, `{color}(...)`, `{% note/diagram %}`, html-сущности) в `ingest/clean.py`.
  Три стратегии чанкинга (`rag.index build --chunking simple|structure|semantic`):
  structure — нода на секцию с хлебными крошками заголовков. Русский стеммер
  (PyStemmer) для BM25 — матчит грамматические формы. Качественно structure
  чище simple; численный замер дельты — в Фазе 6.
- [~] **Фаза 5 — Цитаты/abstention + мультимодальность.** Сделано (A): Pydantic-ответ
  `GroundedAnswer` (answer + citations + abstained) через `structured_predict`;
  цитаты ссылаются на конкретные чанки и сверяются с найденными
  (`unsupported_citations`); abstention на off-topic. Осталось (B): caption PNG
  через Claude vision → текстовый узел.
- [~] **Фаза 6 — Evaluation harness + CI-гейт.** Сделано (Шаг 1): golden-набор
  (`eval/golden/`), детерминированные retrieval-метрики (hit@k/MRR/recall@k/
  precision@k, реализованы явно, **без API**), ablation-раннер `rag.eval`.
  Сделано (Шаг 2): generation-метрики (faithfulness/answer_relevancy/
  abstention) через свой лёгкий Haiku-судью. Числа — в ablation ниже. Осталось:
  CI-гейт на GitHub Actions с порогом регрессии (Шаг 3).
- [ ] **Фаза 7 (опц.) — API/UI + router.** FastAPI + Streamlit; query router и
  text-to-SQL как «фаза 2 источников».

## Ablation study

Главный артефакт проекта — изолированный вклад каждого приёма в числах.

### Retrieval-метрики (Шаг 1, без API)

Детерминированная оценка поиска по golden-набору (`rag.eval`): нашёлся ли нужный
документ и на какой позиции. Прогон на срезе `api/deliveryservice` (77 файлов),
чанкинг `structure`, 14 размеченных вопросов, k=5.

| Конфигурация | hit@5 | MRR | recall@5 | precision@5 |
|---|---|---|---|---|
| dense | 0.86 | **0.77** | 0.86 | 0.56 |
| bm25 | 0.71 | 0.63 | 0.68 | 0.35 |
| hybrid (BM25 + RRF) | 0.86 | 0.72 | 0.82 | 0.45 |
| hybrid + reranker | **0.93** | **0.80** | **0.93** | **0.68** |

Что видно: **reranker выигрывает по всем метрикам**; bm25 в одиночку слабее всех;
**hybrid по MRR/precision чуть хуже dense** (RRF-консенсус размывает точность
dense, когда bm25 не согласен) — и именно reranker поверх это исправляет.

> Воспроизвести: `uv run python -m rag.index build --path <срез> --chunking structure`
> затем `uv run python -m rag.eval --golden deliveryservice.json`.

### Generation-метрики (Шаг 2, свой Haiku-судья)

Оценка **самого ответа** через лёгкий LLM-судью (прозрачный промпт, не ragas).
Прогон: `hybrid + rerank`, chunking=`structure`, генерация и судья — Haiku,
16 вопросов golden-набора.

| Метрика | Значение | Что показывает |
|---|---|---|
| faithfulness | **0.95** | ответ опирается на контекст (прямой прокси галлюцинаций) |
| answer_relevancy | 0.83 | ответ по существу вопроса |
| abstention_accuracy | **0.94** | верно решает отвечать / воздержаться (15/16) |

Из 14 отвечаемых Haiku ответил на 13 (один ошибочно пропустил), оба off-topic
корректно ушли в abstention. context_precision/recall — в retrieval-таблице выше
(recall@k / precision@k).

> Воспроизвести: `uv run python -m rag.eval --golden deliveryservice.json --generation --limit 16`.
> Генерацию можно поднять до Sonnet: `--gen-model claude-sonnet-5` (дороже, abstention-ошибок меньше).

---

## Документация

- [docs/how-it-works.md](docs/how-it-works.md) — карта пайплайна, полный справочник CLI, артефакты на диске
- [docs/setup.md](docs/setup.md) — установка и запуск локально
- [docs/architecture.md](docs/architecture.md) — архитектура и обоснование решений
- [docs/evaluation.md](docs/evaluation.md) — методология оценки и CI-гейт

## Структура репозитория

```
rag-hybrid-serach/
├── data/                 # корпус (в .gitignore); data/README.md — что сюда класть
├── docs/                 # документация (how-it-works, setup, architecture, evaluation)
├── src/rag/              # исходники пакета
│   ├── ingest/           # парсинг Markdown-вики (+ caption картинок — Фаза 5)
│   │   ├── __main__.py   #   CLI инжеста
│   │   └── markdown.py   #   разбор .md: фронтматтер, секции, header_path
│   ├── generation/       # Claude + grounded-промпт (Pydantic-схемы ответа — Фаза 5)
│   │   └── answer.py
│   ├── config.py         # pydantic-settings (.env)
│   ├── embeddings.py     # multilingual-e5-small (CPU)
│   ├── index.py          # сборка индекса: Qdrant (dense) + docstore.json (BM25)
│   ├── retrieval.py      # dense / BM25 / RRF-гибрид / reranker
│   └── query.py          # CLI запроса
├── tests/                # pytest + fixtures/wiki (синтетический корпус)
├── storage/              # индекс Qdrant + docstore (в .gitignore)
├── .env.example          # шаблон конфига (ключ API и т.д.)
├── pyproject.toml        # зависимости по фазам (uv)
└── README.md

# появится позже: eval/ (golden-set + прогоны Ragas, Фаза 6)
```
