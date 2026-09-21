"""Tests for the live web UI server and API endpoints."""

from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient
from notes_qa.ingest import DocumentChunk
from notes_qa.server import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_ui_html_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "notes-qa" in response.text
    assert "Grounded Answer" in response.text


@patch("notes_qa.server.get_chroma_client")
def test_api_status(mock_get_client, client):
    mock_chroma = MagicMock()
    mock_coll = MagicMock()
    mock_coll.count.return_value = 8
    mock_chroma.get_collection.return_value = mock_coll
    mock_get_client.return_value = mock_chroma

    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["indexed_chunks"] == 8


@patch("notes_qa.server.retrieve_chunks")
def test_api_retrieve(mock_retrieve, client):
    chunk = DocumentChunk(
        chunk_id="c1",
        text="Sample text",
        file_path="/notes/doc.md",
        file_name="doc.md",
        doc_type="markdown",
        heading="Intro",
        distance=0.2,
        score=0.85,
    )
    mock_retrieve.return_value = [chunk]

    response = client.post("/api/retrieve", json={"query": "test query", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["chunks"][0]["file_name"] == "doc.md"
    assert data["chunks"][0]["score"] == 0.85


@patch("notes_qa.server.generate_answer")
@patch("notes_qa.server.retrieve_chunks")
def test_api_ask(mock_retrieve, mock_generate, client):
    chunk = DocumentChunk(
        chunk_id="c1",
        text="Rate limiting",
        file_path="/notes/rate.md",
        file_name="rate.md",
        doc_type="markdown",
        heading="Token Bucket",
    )
    mock_retrieve.return_value = [chunk]
    mock_generate.return_value = (
        'Token bucket smooths traffic [rate.md, "Token Bucket"].',
        ['rate.md — "Token Bucket"'],
    )

    response = client.post("/api/ask", json={"question": "rate limiting", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert "Token bucket smooths traffic" in data["answer"]
    assert len(data["sources"]) == 1


def test_api_ingest_not_found(client):
    response = client.post("/api/ingest", json={"folder": "/invalid/nonexistent/path"})
    assert response.status_code == 404


@patch("notes_qa.server.generate_answer")
@patch("notes_qa.server.retrieve_chunks")
def test_api_ask_with_provider(mock_retrieve, mock_generate, client):
    chunk = DocumentChunk(
        chunk_id="c1",
        text="Rate limiting",
        file_path="/notes/rate.md",
        file_name="rate.md",
        doc_type="markdown",
        heading="Token Bucket",
    )
    mock_retrieve.return_value = [chunk]
    mock_generate.return_value = ("Extracted answer", ["source1"])

    response = client.post(
        "/api/ask",
        json={"question": "rate limiting", "top_k": 3, "provider": "offline"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "offline"
    assert data["answer"] == "Extracted answer"

