"""Document loading, chunking, embedding, and storage in Chroma."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    import chromadb
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

from notes_qa.config import (
    COLLECTION_NAME,
    get_db_path,
    get_embedding_model,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents an extracted and chunked segment of a document."""

    chunk_id: str
    text: str
    file_path: str
    file_name: str
    doc_type: str  # 'pdf' or 'markdown'
    page: Optional[int] = None
    heading: Optional[str] = None
    distance: Optional[float] = None
    score: Optional[float] = None

    @property
    def source_citation(self) -> str:
        """Formatted source string for terminal display in Sources section."""
        if self.doc_type == "pdf" and self.page is not None:
            return f"{self.file_name} — page {self.page}"
        elif self.doc_type == "markdown" and self.heading:
            return f'{self.file_name} — "{self.heading}"'
        return self.file_name

    @property
    def inline_citation_tag(self) -> str:
        """Inline citation tag for LLM prompting and inline references."""
        if self.doc_type == "pdf" and self.page is not None:
            return f"[{self.file_name}, p. {self.page}]"
        elif self.doc_type == "markdown" and self.heading:
            return f'[{self.file_name}, "{self.heading}"]'
        return f"[{self.file_name}]"


def split_text_into_chunks(
    text: str,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[str]:
    """Split text into overlapping chunks (~500 tokens with ~50 token overlap).

    Using approx 1 token ≈ 0.75 words (approx 375 words for 500 tokens).
    """
    # 500 tokens ≈ 375-400 words; 50 tokens ≈ 35-40 words
    chunk_size_words = max(50, int(chunk_size_tokens * 0.75))
    overlap_words = max(5, int(overlap_tokens * 0.75))

    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks = []
    start = 0
    step = max(1, chunk_size_words - overlap_words)

    while start < len(words):
        chunk_words = words[start : start + chunk_size_words]
        chunks.append(" ".join(chunk_words))
        start += step
        if start + overlap_words >= len(words):
            break

    return chunks


def parse_markdown_file(file_path: Path) -> list[DocumentChunk]:
    """Parse a Markdown file, preserving heading context for each chunk."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="latin-1", errors="replace")

    chunks: list[DocumentChunk] = []
    lines = content.splitlines()

    current_heading: Optional[str] = None
    current_lines: list[str] = []
    section_index = 0

    heading_regex = re.compile(r"^(#{1,6})\s+(.*)$")

    def flush_section(heading: Optional[str], text_lines: list[str]) -> None:
        nonlocal section_index
        text = "\n".join(text_lines).strip()
        if not text:
            return
        split_segments = split_text_into_chunks(text)
        for sub_idx, segment in enumerate(split_segments):
            chunk_id = f"{file_path.name}_s{section_index}_c{sub_idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=segment,
                    file_path=str(file_path.resolve()),
                    file_name=file_path.name,
                    doc_type="markdown",
                    heading=heading,
                )
            )
        section_index += 1

    for line in lines:
        match = heading_regex.match(line)
        if match:
            if current_lines:
                flush_section(current_heading, current_lines)
                current_lines = []
            current_heading = match.group(2).strip()
        else:
            current_lines.append(line)

    if current_lines:
        flush_section(current_heading, current_lines)

    return chunks


def parse_pdf_file(file_path: Path) -> list[DocumentChunk]:
    """Extract text page-by-page from a PDF file using pypdf."""
    from pypdf import PdfReader

    chunks: list[DocumentChunk] = []
    try:
        reader = PdfReader(str(file_path))
    except Exception as e:
        logger.warning("Failed to parse PDF %s: %s", file_path, e)
        return []

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if not page_text:
            continue

        segments = split_text_into_chunks(page_text)
        for seg_idx, segment in enumerate(segments):
            chunk_id = f"{file_path.name}_p{page_num}_c{seg_idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=segment,
                    file_path=str(file_path.resolve()),
                    file_name=file_path.name,
                    doc_type="pdf",
                    page=page_num,
                )
            )

    return chunks


def get_embedding_function(model_choice: Optional[str] = None) -> Any:
    """Get Chroma embedding function based on configuration."""
    choice = (model_choice or get_embedding_model()).lower()

    if choice == "openai":
        from chromadb.utils import embedding_functions

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required when EMBEDDING_MODEL=openai"
            )
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    elif choice == "anthropic":
        # Anthropic does not have a native embedding API. Notify and fallback to local.
        logger.warning(
            "Anthropic does not offer a standalone embeddings API endpoint. Falling back to local sentence-transformers."
        )

    # Local default: all-MiniLM-L6-v2 via sentence-transformers or chromadb ONNX runtime
    try:
        from chromadb.utils import embedding_functions

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    except Exception:
        from chromadb.utils import embedding_functions

        return embedding_functions.DefaultEmbeddingFunction()


def get_chroma_client(db_path: Optional[str] = None) -> Any:
    """Create or return a persistent Chroma client."""
    import chromadb

    resolved_path = get_db_path(db_path)
    os.makedirs(resolved_path, exist_ok=True)
    return chromadb.PersistentClient(path=resolved_path)


def get_or_create_collection(
    client: Any,
    rebuild: bool = False,
    embedding_fn: Any = None,
) -> Any:
    """Get existing collection or create new one, wiping if rebuild=True."""
    if rebuild:
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

    if embedding_fn is None:
        embedding_fn = get_embedding_function()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def scan_folder(folder_path: Path) -> tuple[list[Path], list[Path], int]:
    """Scan folder recursively for PDFs and Markdown files.

    Returns (pdf_files, md_files, skipped_unsupported_files_count).
    """
    pdf_files: list[Path] = []
    md_files: list[Path] = []
    skipped_count = 0

    for path in folder_path.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext == ".pdf":
            pdf_files.append(path)
        elif ext in [".md", ".markdown"]:
            md_files.append(path)
        else:
            skipped_count += 1

    return sorted(pdf_files), sorted(md_files), skipped_count


def ingest_folder(
    folder: str | Path,
    db_path: Optional[str] = None,
    rebuild: bool = False,
    embedding_fn: Any = None,
) -> dict[str, Any]:
    """Scan folder, extract chunks, embed and store in Chroma vector database."""
    folder_path = Path(folder).resolve()
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder '{folder}' does not exist.")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Path '{folder}' is not a directory.")

    pdf_files, md_files, skipped_count = scan_folder(folder_path)
    total_files = len(pdf_files) + len(md_files)

    if total_files == 0:
        return {
            "folder": str(folder_path),
            "total_files": 0,
            "pdf_count": 0,
            "md_count": 0,
            "skipped_count": skipped_count,
            "chunk_count": 0,
            "elapsed_seconds": 0.0,
            "db_path": get_db_path(db_path),
        }

    chunks: list[DocumentChunk] = []

    for pdf in pdf_files:
        chunks.extend(parse_pdf_file(pdf))

    for md in md_files:
        chunks.extend(parse_markdown_file(md))

    start_time = time.perf_counter()

    client = get_chroma_client(db_path)
    collection = get_or_create_collection(
        client=client,
        rebuild=rebuild,
        embedding_fn=embedding_fn,
    )

    if chunks:
        # Batch insert to avoid collection batch size limits
        batch_size = 200
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "file_path": c.file_path,
                        "file_name": c.file_name,
                        "doc_type": c.doc_type,
                        "page": c.page if c.page is not None else -1,
                        "heading": c.heading or "",
                        "source_citation": c.source_citation,
                        "inline_citation_tag": c.inline_citation_tag,
                    }
                    for c in batch
                ],
            )

    elapsed = time.perf_counter() - start_time
    resolved_db_path = get_db_path(db_path)

    return {
        "folder": str(folder_path),
        "total_files": total_files,
        "pdf_count": len(pdf_files),
        "md_count": len(md_files),
        "skipped_count": skipped_count,
        "chunk_count": len(chunks),
        "elapsed_seconds": elapsed,
        "db_path": resolved_db_path,
    }
