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

## План реализации и прогресс

Легенда: `[ ]` не начато · `[~]` в работе · `[x]` готово

- [x] **Фаза 0 — Скелет проекта.** Репозиторий, зависимости (uv/pyproject,
  установка по фазам), конфиг (`pydantic-settings` + `.env`), каркас пакета
  `src/rag/`, парсер MD-вики с сохранением структуры заголовков + CLI
  `python -m rag.ingest`, фикстуры и тесты (6 зелёных). Qdrant — в
  embedded-режиме (локальная папка), Docker не требуется.
- [ ] **Фаза 1 — Naive RAG baseline.** Только dense + простой чанкинг, ответ
  через Claude end-to-end. → Первые метрики Ragas. **Baseline записан.**
- [ ] **Фаза 2 — Hybrid.** + BM25 + RRF. Замер дельты.
- [ ] **Фаза 3 — Reranker.** + bge-reranker-base. Замер дельты.
- [ ] **Фаза 4 — Чанкинг.** MarkdownNodeParser vs SemanticSplitter. Замер дельты.
- [ ] **Фаза 5 — Мультимодальность + цитаты/abstention.** Caption PNG через
  Claude vision; Pydantic-ответ с grounded-цитатами.
- [ ] **Фаза 6 — Evaluation harness + CI-гейт.** Генерация golden-set через
  Ragas testset generation → ручная чистка ~50–100 Q&A → версионирование в JSON.
  GitHub Actions: eval на подмножестве с Haiku-судьёй, порог регрессии.
- [ ] **Фаза 7 (опц.) — API/UI + router.** FastAPI + Streamlit; query router и
  text-to-SQL как «фаза 2 источников».

## Ablation study

Главный артефакт проекта — таблица заполняется по мере прохождения фаз.

| Конфигурация | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| Naive (dense + simple chunk) | — | — | — | — |
| + Hybrid (BM25 + RRF) | — | — | — | — |
| + Reranker | — | — | — | — |
| + Semantic chunking | — | — | — | — |

---

## Документация

- [docs/architecture.md](docs/architecture.md) — архитектура и обоснование решений
- [docs/setup.md](docs/setup.md) — установка и запуск локально
- [docs/evaluation.md](docs/evaluation.md) — методология оценки и CI-гейт

## Структура репозитория (целевая)

```
rag-hybrid-serach/
├── data/                 # корпус (в .gitignore)
├── docs/                 # документация
├── src/rag/              # исходники пакета
│   ├── ingest/           # парсинг MD, caption картинок, чанкинг
│   ├── retrieval/        # BM25, dense, RRF, reranker
│   ├── generation/       # Claude + Pydantic-схемы ответа
│   └── config.py
├── eval/                 # golden-set + прогоны Ragas
├── .env.example
├── pyproject.toml
└── README.md
```
