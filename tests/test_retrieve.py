"""Tests for similarity retrieval."""

from unittest.mock import MagicMock, patch
from notes_qa.retrieve import retrieve_chunks


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
