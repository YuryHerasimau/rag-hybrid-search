"""Построение и загрузка векторного индекса (Qdrant embedded).

Фаза 1 — naive baseline: один Document на .md-файл, простой чанкинг
SentenceSplitter, плотные эмбеддинги, хранение в Qdrant в embedded-режиме
(локальная папка storage/qdrant, без Docker-сервера).

Фаза 2 — гибрид: те же ноды дополнительно сохраняются в docstore.json, чтобы
их мог поднять лексический BM25-retriever. Важно, что Qdrant и docstore строятся
из ОДНОГО набора нод за один прогон — тогда node_id совпадают и RRF корректно
сливает дубликаты из dense- и BM25-выдачи.

Фаза 4 — чанкинг: три стратегии (simple | structure | semantic), выбор через
build_nodes()/флаг --chunking. По умолчанию simple (baseline Фазы 1).
"""

from __future__ import annotations

import atexit
from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag.config import PROJECT_ROOT, settings
from rag.embeddings import build_embed_model
from rag.ingest import load_wiki

STORAGE_ROOT = PROJECT_ROOT / "storage"
STORAGE_DIR = STORAGE_ROOT / "qdrant"
# Ноды для BM25 (лексический поиск не хранится в Qdrant — ему нужен сырой текст)
DOCSTORE_PATH = STORAGE_ROOT / "docstore.json"


# Embedded-Qdrant держит эксклюзивный файловый лок на папку хранилища: в одном
# процессе допустим только ОДИН QdrantClient на путь. Поэтому клиент — синглтон,
# иначе второй load_index() в том же процессе (например dense, а затем hybrid)
# падает с «Storage folder is already accessed by another instance».
_qdrant_client: QdrantClient | None = None


def _client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(STORAGE_DIR))
        # чистое закрытие на выходе: освобождает лок (иначе следующий процесс
        # может упереться в stale-lock) и глушит шум msvcrt в QdrantClient.__del__
        atexit.register(_close_client)
    return _qdrant_client


def _close_client() -> None:
    global _qdrant_client
    if _qdrant_client is not None:
        _qdrant_client.close()
        _qdrant_client = None


CHUNKING_STRATEGIES = ("simple", "structure", "semantic")


def to_documents(wiki_docs) -> list[Document]:
    """WikiDocument -> llama_index.Document с метаданными для цитирования."""
    return [
        Document(
            text=d.text,
            metadata={"rel_path": d.rel_path, "title": d.title},
            doc_id=d.rel_path,
        )
        for d in wiki_docs
    ]


def _simple_nodes(wiki_docs, chunk_size: int, chunk_overlap: int) -> list[BaseNode]:
    """Фаза 1 baseline: один Document на файл, слепой SentenceSplitter."""
    docs = to_documents(wiki_docs)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents(docs, show_progress=True)


def _structure_nodes(wiki_docs, chunk_size: int, chunk_overlap: int) -> list[BaseNode]:
    """Structure-aware: по одной секции вики (заголовок + тело) на Document.

    Хлебную крошку заголовков дописываем ПРЕФИКСОМ в текст — тогда чанк
    самодостаточен и для эмбеддингов, и для BM25 (он метаданные не видит), и
    цитата сразу читается. Крупные секции добивает SentenceSplitter, мелкие
    остаются целыми (в отличие от simple, который режет вслепую по размеру).
    """
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    section_docs: list[Document] = []
    for d in wiki_docs:
        for sec in d.sections:
            crumb = sec.breadcrumb
            text = f"{crumb}\n\n{sec.text}" if sec.header_path else sec.text
            doc = Document(
                text=text,
                metadata={"rel_path": d.rel_path, "title": d.title, "breadcrumb": crumb},
            )
            # крошка уже в тексте — не дублируем её в эмбеддинге/промпте
            doc.excluded_embed_metadata_keys = ["breadcrumb"]
            doc.excluded_llm_metadata_keys = ["breadcrumb"]
            section_docs.append(doc)
    return splitter.get_nodes_from_documents(section_docs, show_progress=True)


def _semantic_nodes(wiki_docs) -> list[BaseNode]:
    """Семантический сплиттинг: границы чанков там, где «проседает» близость
    соседних предложений (считается эмбеддингами). На CPU медленно — для
    ablation-сравнения, не для продакшена на слабом ноуте."""
    from llama_index.core.node_parser import SemanticSplitterNodeParser

    docs = to_documents(wiki_docs)
    parser = SemanticSplitterNodeParser(
        embed_model=build_embed_model(),
        breakpoint_percentile_threshold=95,
    )
    return parser.get_nodes_from_documents(docs, show_progress=True)


def build_nodes(
    wiki_docs,
    chunking: str = "simple",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[BaseNode]:
    """Диспетчер стратегий чанкинга (ablation, Фаза 4)."""
    if chunking == "simple":
        return _simple_nodes(wiki_docs, chunk_size, chunk_overlap)
    if chunking == "structure":
        return _structure_nodes(wiki_docs, chunk_size, chunk_overlap)
    if chunking == "semantic":
        return _semantic_nodes(wiki_docs)
    raise ValueError(f"Неизвестная стратегия чанкинга: {chunking!r} ({'|'.join(CHUNKING_STRATEGIES)})")


def build_index(
    path: str | Path,
    chunking: str = "simple",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    caption: bool = False,
) -> VectorStoreIndex:
    """Строит индекс с нуля (пересоздаёт коллекцию).

    Чанкинг делаем ЯВНО (не через transformations), чтобы получить один список
    нод и использовать его дважды: для эмбеддингов в Qdrant и для docstore
    (BM25). Так node_id гарантированно совпадают между dense- и BM25-выдачей.

    caption=True (Фаза 5B) — локальные картинки страниц описываются через Claude
    vision и индексируются как текст (с кэшем, нужен ANTHROPIC_API_KEY).
    """
    if caption:
        from rag.ingest.images import CaptionCache, build_vision_captioner

        wiki = load_wiki(path, captioner=build_vision_captioner(), cache=CaptionCache())
    else:
        wiki = load_wiki(path)
    nodes = build_nodes(wiki, chunking, chunk_size, chunk_overlap)

    # 1) Сохраняем ноды на диск — их поднимет лексический BM25 (Фаза 2)
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    docstore.persist(str(DOCSTORE_PATH))

    # 2) Эмбеддим те же ноды и кладём в Qdrant (dense-поиск)
    client = _client()
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)

    vector_store = QdrantVectorStore(client=client, collection_name=settings.qdrant_collection)
    storage = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage,
        embed_model=build_embed_model(),
        show_progress=True,
    )
    _close_client()  # освобождаем лок сразу после сборки (не ждём выхода)
    return index


def load_nodes() -> list[BaseNode]:
    """Поднимает сохранённые ноды из docstore (для BM25-retriever)."""
    if not DOCSTORE_PATH.exists():
        raise FileNotFoundError(
            "docstore.json не найден. Пересобери индекс: uv run python -m rag.index build"
        )
    docstore = SimpleDocumentStore.from_persist_path(str(DOCSTORE_PATH))
    return list(docstore.docs.values())


def load_index() -> VectorStoreIndex:
    """Открывает существующий индекс для запросов."""
    client = _client()
    if not client.collection_exists(settings.qdrant_collection):
        client.close()
        raise FileNotFoundError(
            "Индекс не найден. Сначала собери его: uv run python -m rag.index build"
        )
    vector_store = QdrantVectorStore(client=client, collection_name=settings.qdrant_collection)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=build_embed_model())


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="rag.index", description="Сборка векторного индекса")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="собрать индекс с нуля")
    build.add_argument("--path", type=Path, default=settings.raw_dir, help="каталог с .md")
    build.add_argument(
        "--chunking",
        choices=CHUNKING_STRATEGIES,
        default="simple",
        help="стратегия чанкинга (ablation, Фаза 4): simple | structure | semantic",
    )
    build.add_argument("--chunk-size", type=int, default=512)
    build.add_argument("--chunk-overlap", type=int, default=64)
    build.add_argument(
        "--captions",
        action="store_true",
        help="описывать локальные картинки через Claude vision и индексировать (Фаза 5B, нужен ключ)",
    )
    args = parser.parse_args()

    if args.cmd == "build":
        cap = " + captions" if args.captions else ""
        print(f"Строю индекс из {args.path} (чанкинг: {args.chunking}{cap}) ...")
        build_index(
            args.path,
            chunking=args.chunking,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            caption=args.captions,
        )
        print(f"Готово. Индекс в {STORAGE_DIR}")


if __name__ == "__main__":
    _main()
