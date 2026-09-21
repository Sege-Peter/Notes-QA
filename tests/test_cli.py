"""Tests for Click CLI commands."""

from unittest.mock import MagicMock, patch
from click.testing import CliRunner
from notes_qa.cli import main
from notes_qa.ingest import DocumentChunk


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "notes-qa" in result.output
    assert "ingest" in result.output
    assert "ask" in result.output


@patch("notes_qa.cli.ingest_folder")
def test_cli_ingest(mock_ingest, tmp_path):
    mock_ingest.return_value = {
        "folder": str(tmp_path),
        "total_files": 14,
        "pdf_count": 9,
        "md_count": 5,
        "skipped_count": 0,
        "chunk_count": 342,
        "elapsed_seconds": 12.4,
        "db_path": "./.db",
    }

    runner = CliRunner()
    result = runner.invoke(main, ["ingest", str(tmp_path)])
    assert result.exit_code == 0
    assert f"Scanning {tmp_path}" in result.output
    assert "Found 14 files (9 PDFs, 5 Markdown)" in result.output
    assert "Chunked into 342 segments" in result.output
    assert "Embedding... done in 12.4s" in result.output
    assert "Stored in ./.db" in result.output


@patch("notes_qa.cli.ingest_folder")
def test_cli_ingest_empty_folder(mock_ingest, tmp_path):
    mock_ingest.return_value = {
        "folder": str(tmp_path),
        "total_files": 0,
        "pdf_count": 0,
        "md_count": 0,
        "skipped_count": 2,
        "chunk_count": 0,
        "elapsed_seconds": 0.0,
        "db_path": "./.db",
    }

    runner = CliRunner()
    result = runner.invoke(main, ["ingest", str(tmp_path)])
    assert result.exit_code == 0
    assert "No supported documents found" in result.output
    assert "Skipped 2 unsupported files" in result.output


@patch("notes_qa.cli.generate_answer")
@patch("notes_qa.cli.retrieve_chunks")
def test_cli_ask(mock_retrieve, mock_generate):
    mock_chunk = DocumentChunk(
        chunk_id="c1",
        text="Rate limiting info",
        file_path="/notes/rate-limiting-notes.md",
        file_name="rate-limiting-notes.md",
        doc_type="markdown",
        heading="Token Bucket",
    )
    mock_retrieve.return_value = [mock_chunk]
    mock_generate.return_value = (
        'Token-bucket rate limiting smooths traffic [rate-limiting-notes.md, "Token Bucket"].',
        ['rate-limiting-notes.md — "Token Bucket"'],
    )

    runner = CliRunner()
    result = runner.invoke(main, ["ask", "What did I write about rate limiting?"])
    assert result.exit_code == 0
    assert "Answer:" in result.output
    assert (
        'Token-bucket rate limiting smooths traffic [rate-limiting-notes.md, "Token Bucket"].'
        in result.output
    )
    assert "Sources:" in result.output
    assert '  - rate-limiting-notes.md — "Token Bucket"' in result.output


@patch("notes_qa.cli.retrieve_chunks")
def test_cli_ask_empty_or_irrelevant(mock_retrieve):
    mock_retrieve.return_value = []
    runner = CliRunner()
    result = runner.invoke(main, ["ask", "What did I write?"])
    assert result.exit_code == 0
    assert "No relevant information found in your notes for this question" in result.output
