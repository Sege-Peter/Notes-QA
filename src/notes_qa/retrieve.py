"""Similarity search over the Chroma vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from notes_qa.config import COLLECTION_NAME, get_db_path
from notes_qa.ingest import DocumentChunk, get_chroma_client, get_embedding_function


def retrieve_chunks(
    query: str,
    top_k: int = 4,
    db_path: Optional[str] = None,
    embedding_fn: Any = None,
) -> list[DocumentChunk]:
    """Retrieve the top-k most relevant document chunks for a given query."""
    resolved_db_path = get_db_path(db_path)
    if not Path(resolved_db_path).exists():
        return []

    client = get_chroma_client(db_path=resolved_db_path)

    if embedding_fn is None:
        try:
            embedding_fn = get_embedding_function()
        except Exception:
            embedding_fn = None

    try:
        if embedding_fn is not None:
            collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_fn,
            )
        else:
            collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        # Collection does not exist yet
        return []

    total_count = collection.count()
    if total_count == 0:
        return []

    actual_k = min(top_k, total_count)
    results = collection.query(
        query_texts=[query],
        n_results=actual_k,
    )

    chunks: list[DocumentChunk] = []

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    for doc_id, text, meta in zip(ids, docs, metas):
        page_val = meta.get("page")
        page = int(page_val) if page_val is not None and page_val != -1 else None
        heading_val = meta.get("heading") or None

        chunks.append(
            DocumentChunk(
                chunk_id=doc_id,
                text=text,
                file_path=meta.get("file_path", ""),
                file_name=meta.get("file_name", ""),
                doc_type=meta.get("doc_type", "markdown"),
                page=page,
                heading=heading_val,
            )
        )

    return chunks
