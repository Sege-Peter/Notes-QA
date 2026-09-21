"""Tests for similarity retrieval, hybrid BM25 search, and threshold filtering."""

from unittest.mock import MagicMock, patch
from notes_qa.retrieve import SimpleBM25, retrieve_chunks, tokenize


def test_tokenize():
    text = "Token-bucket, rate limiting (v2.0)!"
    tokens = tokenize(text)
    assert "token" in tokens
    assert "bucket" in tokens
    assert "limiting" in tokens


def test_simple_bm25():
    corpus = [
        "Token bucket rate limiting smooths traffic bursts",
        "Redis INCR and EXPIRE provides fixed window limiting",
        "PostgreSQL multi version concurrency control MVCC",
    ]
    bm25 = SimpleBM25(corpus)
    score_token = bm25.score("token bucket", 0)
    score_other = bm25.score("token bucket", 2)
    assert score_token > score_other
    assert score_other == 0.0


def test_retrieve_chunks_empty_path(tmp_path):
    empty_dir = tmp_path / "non_existent_db"
    chunks = retrieve_chunks("test query", top_k=3, db_path=str(empty_dir))
    assert chunks == []


@patch("notes_qa.retrieve.get_chroma_client")
def test_retrieve_chunks_from_collection(mock_get_client, tmp_path):
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.get_collection.return_value = mock_collection

    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        "ids": [["c1", "c2"]],
        "documents": [["Token bucket rate limiting", "Redis INCR limiter"]],
        "distances": [[0.3, 0.4]],
        "metadatas": [
            [
                {
                    "file_name": "rate-limiting.md",
                    "file_path": "/notes/rate-limiting.md",
                    "doc_type": "markdown",
                    "heading": "Token Bucket",
                    "page": -1,
                },
                {
                    "file_name": "system-design.pdf",
                    "file_path": "/notes/system-design.pdf",
                    "doc_type": "pdf",
                    "heading": "",
                    "page": 12,
                },
            ]
        ],
    }

    chunks = retrieve_chunks("rate limiting", top_k=2, db_path=str(db_dir))
    assert len(chunks) == 2
    assert chunks[0].file_name == "rate-limiting.md"
    assert chunks[0].heading == "Token Bucket"
    assert chunks[0].source_citation == 'rate-limiting.md — "Token Bucket"'

    assert chunks[1].file_name == "system-design.pdf"
    assert chunks[1].page == 12
    assert chunks[1].source_citation == "system-design.pdf — page 12"


@patch("notes_qa.retrieve.get_chroma_client")
def test_retrieve_chunks_threshold_filtering(mock_get_client, tmp_path):
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.get_collection.return_value = mock_collection

    mock_collection.count.return_value = 1
    # Very high distance (0.95), completely unrelated
    mock_collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["B-Tree page structure and leaf nodes"]],
        "distances": [[0.95]],
        "metadatas": [
            [
                {
                    "file_name": "db.md",
                    "file_path": "/notes/db.md",
                    "doc_type": "markdown",
                    "heading": "B-Tree",
                    "page": -1,
                }
            ]
        ],
    }

    chunks = retrieve_chunks(
        "Who won the 1998 World Cup?",
        top_k=2,
        db_path=str(db_dir),
        distance_threshold=0.85,
    )
    assert chunks == []
