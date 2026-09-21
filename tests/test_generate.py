"""Tests for prompt construction, LLM generation, and formatting."""

from unittest.mock import MagicMock
import pytest
from notes_qa.generate import (
    build_context_block,
    format_qa_output,
    generate_answer,
    get_unique_sources,
)
from notes_qa.ingest import DocumentChunk


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="c1",
            text="Token-bucket rate limiting is preferable to fixed windows.",
            file_path="/notes/rate-limiting.md",
            file_name="rate-limiting.md",
            doc_type="markdown",
            heading="Token Bucket",
        ),
        DocumentChunk(
            chunk_id="c2",
            text="Redis INCR + EXPIRE is common for fixed window.",
            file_path="/notes/system-design.pdf",
            file_name="system-design.pdf",
            doc_type="pdf",
            page=12,
        ),
    ]


def test_build_context_block(sample_chunks):
    context = build_context_block(sample_chunks)
    assert 'Section: "Token Bucket"' in context
    assert "Page: 12" in context
    assert '[rate-limiting.md, "Token Bucket"]' in context
    assert "[system-design.pdf, p. 12]" in context


def test_get_unique_sources(sample_chunks):
    sources = get_unique_sources(sample_chunks)
    assert sources == [
        'rate-limiting.md — "Token Bucket"',
        "system-design.pdf — page 12",
    ]


def test_format_qa_output():
    ans = "This is the answer [doc.md, \"Section\"]."
    sources = ['doc.md — "Section"']
    output = format_qa_output(ans, sources)

    assert "Answer:" in output
    assert "This is the answer [doc.md, \"Section\"]." in output
    assert "Sources:" in output
    assert '  - doc.md — "Section"' in output


def test_generate_answer_empty_chunks():
    ans, sources = generate_answer("test query", [])
    assert (
        "no relevant information found" in ans.lower()
        or "could not find" in ans.lower()
    )
    assert sources == []


def test_generate_answer_missing_api_key(sample_chunks, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        generate_answer("query", sample_chunks, api_key=None, client=None)


def test_generate_answer_mock_client(sample_chunks):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.text = 'Token bucket smooths traffic [rate-limiting.md, "Token Bucket"].'
    mock_response.content = [mock_block]
    mock_client.messages.create.return_value = mock_response

    ans, sources = generate_answer(
        query="Explain token bucket",
        chunks=sample_chunks,
        client=mock_client,
    )

    assert 'Token bucket smooths traffic [rate-limiting.md, "Token Bucket"].' in ans
    assert 'rate-limiting.md — "Token Bucket"' in sources
    assert "system-design.pdf — page 12" in sources


def test_generate_answer_gemini_missing_api_key(sample_chunks, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        generate_answer("query", sample_chunks, provider="gemini", api_key=None, client=None)


def test_generate_answer_gemini_mock_client(sample_chunks):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = 'Token bucket is preferred for bursty traffic [rate-limiting.md, "Token Bucket"].'
    mock_client.models.generate_content.return_value = mock_response

    ans, sources = generate_answer(
        query="Explain token bucket",
        chunks=sample_chunks,
        client=mock_client,
        provider="gemini",
    )

    assert "Token bucket is preferred for bursty traffic" in ans
    assert 'rate-limiting.md — "Token Bucket"' in sources


def test_generate_answer_offline_zero_api_key(sample_chunks):
    ans, sources = generate_answer(
        query="Explain token bucket rate limiting",
        chunks=sample_chunks,
        provider="offline",
    )

    assert "Based on your notes:" in ans
    assert '[rate-limiting.md, "Token Bucket"]' in ans
    assert len(sources) > 0
    assert 'rate-limiting.md — "Token Bucket"' in sources


def test_generate_answer_offline_irrelevant(sample_chunks):
    ans, sources = generate_answer(
        query="Who was Cleopatra?",
        chunks=sample_chunks,
        provider="offline",
    )
    assert "No relevant information found in your notes for this question." in ans
    assert sources == []

