"""End-to-end integration tests for ingest and retrieve."""

import hashlib
from pathlib import Path
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from notes_qa.ingest import ingest_folder
from notes_qa.retrieve import retrieve_chunks


class FastDeterministicEmbedding(EmbeddingFunction):
    """Fast, offline, deterministic non-zero embedding function for tests."""

    def __init__(self) -> None:
        pass

    def name(self) -> str:
        return "fast_deterministic"

    def __call__(self, input: Documents) -> Embeddings:
        results: Embeddings = []
        for text in input:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [(b / 255.0) for b in h] * 12
            vec = vec[:384]
            norm = sum(x * x for x in vec) ** 0.5
            results.append([x / norm for x in vec])
        return results


def test_end_to_end_ingest_and_retrieve(tmp_path: Path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    db_dir = tmp_path / "db"

    note_file = notes_dir / "rate-limiting-notes.md"
    note_file.write_text(
        """# Distributed Systems

## Token Bucket
Token-bucket rate limiting is preferable to fixed windows because it smooths bursty traffic without rejecting legitimate spikes.

## Fixed Window
Redis INCR + EXPIRE is a common way to implement a simple fixed-window limiter.
""",
        encoding="utf-8",
    )

    embed_fn = FastDeterministicEmbedding()

    stats = ingest_folder(
        folder=notes_dir,
        db_path=str(db_dir),
        rebuild=True,
        embedding_fn=embed_fn,
    )
    assert stats["total_files"] == 1
    assert stats["md_count"] == 1
    assert stats["chunk_count"] >= 2

    # Query similarity search
    chunks = retrieve_chunks(
        query="Token Bucket",
        top_k=2,
        db_path=str(db_dir),
        embedding_fn=embed_fn,
        distance_threshold=2.0,
    )

    assert len(chunks) > 0
    file_names = [c.file_name for c in chunks]
    assert "rate-limiting-notes.md" in file_names
