"""Similarity and hybrid search over the Chroma vector store."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from notes_qa.config import (
    COLLECTION_NAME,
    DEFAULT_SIMILARITY_DISTANCE_THRESHOLD,
    get_db_path,
)
from notes_qa.ingest import DocumentChunk, get_chroma_client, get_embedding_function


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer for BM25 keyword matching."""
    return re.findall(r"\b\w+\b", text.lower())


class SimpleBM25:
    """Lightweight in-memory BM25 ranker for hybrid keyword search."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(tokenize(doc)) for doc in corpus]
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 1.0
        )
        self.doc_freqs: dict[str, int] = Counter()
        self.term_freqs: list[dict[str, int]] = []

        for doc in corpus:
            tokens = tokenize(doc)
            freqs = Counter(tokens)
            self.term_freqs.append(freqs)
            for token in freqs.keys():
                self.doc_freqs[token] += 1

    def score(self, query: str, doc_index: int) -> float:
        """Compute BM25 score of document at doc_index for the given query."""
        if self.corpus_size == 0 or doc_index >= self.corpus_size:
            return 0.0

        query_tokens = tokenize(query)
        doc_len = self.doc_lengths[doc_index]
        freqs = self.term_freqs[doc_index]
        score = 0.0

        for token in query_tokens:
            if token not in freqs:
                continue
            tf = freqs[token]
            df = self.doc_freqs.get(token, 0)
            # Standard Lucene / Robertson IDF
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_length))
            score += idf * (numerator / denominator)

        return score


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    db_path: Optional[str] = None,
    embedding_fn: Any = None,
    distance_threshold: float = DEFAULT_SIMILARITY_DISTANCE_THRESHOLD,
    enable_hybrid: bool = True,
) -> list[DocumentChunk]:
    """Retrieve top-k relevant document chunks using hybrid vector + BM25 search.

    Filters out chunks that fall outside the similarity distance threshold.
    """
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

    # Retrieve a slightly wider pool of candidates for hybrid reranking
    candidate_k = min(total_count, max(top_k * 3, 10))
    results = collection.query(
        query_texts=[query],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(docs)

    candidate_chunks: list[DocumentChunk] = []

    for doc_id, text, meta, dist in zip(ids, docs, metas, distances):
        page_val = meta.get("page")
        page = int(page_val) if page_val is not None and page_val != -1 else None
        heading_val = meta.get("heading") or None

        chunk = DocumentChunk(
            chunk_id=doc_id,
            text=text,
            file_path=meta.get("file_path", ""),
            file_name=meta.get("file_name", ""),
            doc_type=meta.get("doc_type", "markdown"),
            page=page,
            heading=heading_val,
            distance=float(dist) if dist is not None else None,
        )
        candidate_chunks.append(chunk)

    if not candidate_chunks:
        return []

    # Hybrid search: Combine vector similarity with BM25 keyword score
    if enable_hybrid and len(candidate_chunks) > 1:
        corpus = [c.text for c in candidate_chunks]
        bm25 = SimpleBM25(corpus)
        bm25_scores = [bm25.score(query, idx) for idx in range(len(candidate_chunks))]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        for idx, chunk in enumerate(candidate_chunks):
            # Vector similarity: 1 - cosine_distance (bounded in [0, 1])
            vec_sim = max(0.0, min(1.0, 1.0 - (chunk.distance if chunk.distance is not None else 0.5)))
            norm_bm25 = bm25_scores[idx] / max_bm25 if max_bm25 > 0 else 0.0

            # 70% vector semantic similarity + 30% exact keyword BM25
            hybrid_score = 0.7 * vec_sim + 0.3 * norm_bm25
            chunk.score = hybrid_score

        # Sort by combined hybrid score descending
        candidate_chunks.sort(key=lambda c: c.score or 0.0, reverse=True)
    else:
        for chunk in candidate_chunks:
            chunk.score = max(0.0, 1.0 - (chunk.distance if chunk.distance is not None else 0.5))

    # Apply distance / similarity threshold: filter out irrelevant chunks
    filtered_chunks: list[DocumentChunk] = []
    for chunk in candidate_chunks:
        # Keep chunk if distance is within threshold OR if it scored keyword matches
        if chunk.distance is not None and chunk.distance > distance_threshold:
            # If distance exceeds threshold, only include if strong keyword match
            if chunk.score and chunk.score > 0.4:
                filtered_chunks.append(chunk)
        else:
            filtered_chunks.append(chunk)

    return filtered_chunks[:top_k]
