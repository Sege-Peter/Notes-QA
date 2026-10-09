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



def test_api_upload_rejects_unsupported_files(client):
    response = client.post(
        "/api/upload",
        files=[("files", ("evil.exe", b"binary", "application/octet-stream"))],
    )
    assert response.status_code == 400


def test_ui_escapes_untrusted_content(client):
    response = client.get("/")
    assert "function escapeHtml" in response.text
    assert "${escapeHtml(chunk.text)}" in response.text


def test_ui_works_without_cdn_assets(client):
    # The offline mode must not hard-depend on CDN-hosted scripts.
    html = client.get("/").text
    assert "if (window.tailwind)" in html
    assert "if (window.marked)" in html
    assert "marked.parse(data.answer" not in html


# --- public (hosted demo) mode ---

@pytest.fixture
def public_client(tmp_path, monkeypatch):
    import notes_qa.server as server

    samples = tmp_path / "sample_notes"
    samples.mkdir()
    uploads = tmp_path / "uploaded_notes"
    monkeypatch.setattr(server, "PUBLIC_MODE", True)
    monkeypatch.setattr(server, "SAMPLE_NOTES_DIR", samples.resolve())
    monkeypatch.setattr(server, "UPLOAD_DIR", uploads.resolve())
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr(server, "ingest_folder", MagicMock(return_value={"chunks": 0}))
    return TestClient(server.create_app()), server, samples


def test_public_mode_blocks_ingesting_other_folders(public_client, tmp_path):
    client, server, _ = public_client
    outside = tmp_path / "server_files"
    outside.mkdir()
    response = client.post("/api/ingest", json={"folder": str(outside)})
    assert response.status_code == 403
    server.ingest_folder.assert_not_called()


def test_public_mode_allows_samples_but_never_rebuilds(public_client):
    client, server, samples = public_client
    response = client.post("/api/ingest", json={"folder": str(samples), "rebuild": True})
    assert response.status_code == 200
    assert server.ingest_folder.call_args.kwargs["rebuild"] is False


def test_public_mode_rejects_unsupported_upload(public_client):
    client, server, _ = public_client
    response = client.post("/api/upload", files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))])
    assert response.status_code == 400
    server.ingest_folder.assert_not_called()


def test_public_mode_rejects_oversized_upload(public_client):
    client, server, _ = public_client
    response = client.post("/api/upload", files=[("files", ("big.md", b"#" * 4096, "text/markdown"))])
    assert response.status_code == 413
    assert not (server.UPLOAD_DIR / "big.md").exists()
    server.ingest_folder.assert_not_called()


def test_public_mode_accepts_small_markdown(public_client):
    client, server, _ = public_client
    response = client.post("/api/upload?rebuild=true", files=[("files", ("note.md", b"# Hi\nhello", "text/markdown"))])
    assert response.status_code == 200
    assert (server.UPLOAD_DIR / "note.md").exists()
    assert server.ingest_folder.call_args.kwargs["rebuild"] is False
