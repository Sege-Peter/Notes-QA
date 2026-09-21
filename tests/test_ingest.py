"""Tests for document loading and chunking."""

from pathlib import Path
import pytest
from notes_qa.ingest import (
    DocumentChunk,
    ingest_folder,
    parse_markdown_file,
    parse_pdf_file,
    scan_folder,
    split_text_into_chunks,
)


def test_split_text_into_chunks():
    # Short text
    text = "Word1 Word2 Word3"
    chunks = split_text_into_chunks(text, chunk_size_tokens=10, overlap_tokens=2)
    assert len(chunks) == 1
    assert chunks[0] == "Word1 Word2 Word3"

    # Longer text requiring chunking
    long_text = " ".join([f"token_{i}" for i in range(600)])
    chunks = split_text_into_chunks(long_text, chunk_size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    assert chunks[0].startswith("token_0")


def test_parse_markdown_file(tmp_path: Path):
    md_file = tmp_path / "rate-limiting.md"
    md_file.write_text(
        """# System Design

Overview of architecture.

## Token Bucket

Token-bucket rate limiting is preferable to fixed windows because it smooths bursty traffic.

## Leaky Bucket

Leaky bucket enforces a steady output rate.
""",
        encoding="utf-8",
    )

    chunks = parse_markdown_file(md_file)
    assert len(chunks) >= 3

    headings = [c.heading for c in chunks]
    assert "System Design" in headings
    assert "Token Bucket" in headings
    assert "Leaky Bucket" in headings

    token_chunk = next(c for c in chunks if c.heading == "Token Bucket")
    assert "bursty traffic" in token_chunk.text
    assert token_chunk.source_citation == 'rate-limiting.md — "Token Bucket"'
    assert token_chunk.inline_citation_tag == '[rate-limiting.md, "Token Bucket"]'


def test_scan_folder(tmp_path: Path):
    (tmp_path / "doc1.md").write_text("# Doc 1", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "doc2.markdown").write_text("# Doc 2", encoding="utf-8")
    (tmp_path / "sub" / "notes.pdf").write_bytes(b"%PDF-1.4 dummy")
    (tmp_path / "sub" / "ignore.txt").write_text("ignore", encoding="utf-8")

    pdfs, mds, skipped = scan_folder(tmp_path)
    assert len(pdfs) == 1
    assert len(mds) == 2
    assert skipped == 1
    assert pdfs[0].name == "notes.pdf"
    md_names = [m.name for m in mds]
    assert "doc1.md" in md_names
    assert "doc2.markdown" in md_names


def test_ingest_empty_folder(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    stats = ingest_folder(empty_dir)
    assert stats["total_files"] == 0
    assert stats["chunk_count"] == 0


def test_ingest_nonexistent_folder():
    with pytest.raises(FileNotFoundError):
        ingest_folder("/non/existent/path/for/sure")


def test_document_chunk_properties():
    pdf_chunk = DocumentChunk(
        chunk_id="pdf_1",
        text="Sample text",
        file_path="/path/to/system-design.pdf",
        file_name="system-design.pdf",
        doc_type="pdf",
        page=12,
    )
    assert pdf_chunk.source_citation == "system-design.pdf — page 12"
    assert pdf_chunk.inline_citation_tag == "[system-design.pdf, p. 12]"

    md_chunk = DocumentChunk(
        chunk_id="md_1",
        text="Sample text",
        file_path="/path/to/notes.md",
        file_name="notes.md",
        doc_type="markdown",
        heading="Architecture",
    )
    assert md_chunk.source_citation == 'notes.md — "Architecture"'
    assert md_chunk.inline_citation_tag == '[notes.md, "Architecture"]'
