"""Построение и загрузка векторного индекса (Qdrant embedded).

Фаза 1 — naive baseline: один Document на .md-файл, простой чанкинг
SentenceSplitter, плотные эмбеддинги, хранение в Qdrant в embedded-режиме
(локальная папка storage/qdrant, без Docker-сервера).

Structure-aware чанкинг подключим в Фазе 4; здесь намеренно самый простой.
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag.config import PROJECT_ROOT, settings
from rag.embeddings import build_embed_model
from rag.ingest import load_wiki

STORAGE_DIR = PROJECT_ROOT / "storage" / "qdrant"


def _client() -> QdrantClient:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(STORAGE_DIR))


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


def build_index(
    path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> VectorStoreIndex:
    """Строит индекс с нуля (пересоздаёт коллекцию)."""
    wiki = load_wiki(path)
    docs = to_documents(wiki)

    client = _client()
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)

    vector_store = QdrantVectorStore(client=client, collection_name=settings.qdrant_collection)
    storage = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage,
        embed_model=build_embed_model(),
        transformations=[splitter],
        show_progress=True,
    )
    client.close()
    return index


def load_index() -> VectorStoreIndex:
    """Открывает существующий индекс для запросов."""
    client = _client()
    if not client.collection_exists(settings.qdrant_collection):
        client.close()
        raise FileNotFoundError(
            "Индекс не найден. Сначала собери его: python -m rag.index build"
        )
    vector_store = QdrantVectorStore(client=client, collection_name=settings.qdrant_collection)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=build_embed_model())


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="rag.index", description="Сборка векторного индекса")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="собрать индекс с нуля")
    build.add_argument("--path", type=Path, default=settings.raw_dir, help="каталог с .md")
    build.add_argument("--chunk-size", type=int, default=512)
    build.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    if args.cmd == "build":
        print(f"Строю индекс из {args.path} ...")
        build_index(args.path, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        print(f"Готово. Индекс в {STORAGE_DIR}")


if __name__ == "__main__":
    _main()
