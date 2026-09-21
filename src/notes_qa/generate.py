"""Prompt construction, LLM generation, and citation formatting using Claude."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

from notes_qa.config import DEFAULT_ANTHROPIC_MODEL, get_anthropic_api_key
from notes_qa.ingest import DocumentChunk


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the user's personal notes and documents.

Rules:
1. Answer ONLY based on the facts provided in the Context below. Do NOT assume, extrapolate, or use outside knowledge.
2. If the provided context does not contain enough information to answer the question, state:
   "No relevant information found in your notes for this question."
3. Cite your sources inline immediately following relevant claims, using the citation tags indicated for each excerpt (for example: [notes.md, "Section Name"] or [document.pdf, p. 12]).
4. Keep the answer clear, grounded, and concise."""


def build_context_block(chunks: list[DocumentChunk]) -> str:
    """Format chunks into a structured context block for the LLM."""
    sections: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        lines = [
            f"--- Excerpt {idx} ---",
            f"File: {chunk.file_name}",
        ]
        if chunk.doc_type == "pdf" and chunk.page is not None:
            lines.append(f"Page: {chunk.page}")
        elif chunk.doc_type == "markdown" and chunk.heading:
            lines.append(f'Section: "{chunk.heading}"')

        lines.append(f"Inline Citation: {chunk.inline_citation_tag}")
        lines.append("Content:")
        lines.append(chunk.text.strip())
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def get_unique_sources(chunks: list[DocumentChunk]) -> list[str]:
    """Return ordered list of unique formatted sources from chunks."""
    seen = set()
    sources: list[str] = []
    for chunk in chunks:
        cite = chunk.source_citation
        if cite and cite not in seen:
            seen.add(cite)
            sources.append(cite)
    return sources


def format_qa_output(answer: str, sources: list[str]) -> str:
    """Format the final answer and sources matching the README output style."""
    output_lines = [
        "Answer:",
        answer.strip(),
    ]

    if sources:
        output_lines.append("")
        output_lines.append("Sources:")
        for src in sources:
            output_lines.append(f"  - {src}")

    return "\n".join(output_lines)


def generate_answer(
    query: str,
    chunks: list[DocumentChunk],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    client: Any = None,
) -> tuple[str, list[str]]:
    """Generate grounded answer from Claude with inline citations and source list."""
    if not chunks:
        return ("No relevant information found in your notes for this question.", [])

    resolved_key = api_key or get_anthropic_api_key()
    if not resolved_key and client is None:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Please add it to your .env file or environment."
        )

    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=resolved_key)

    chosen_model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    context_text = build_context_block(chunks)

    user_message = (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Provide your grounded answer with inline citations:"
    )

    response = client.messages.create(
        model=chosen_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message},
        ],
    )

    # Extract text from message response
    answer_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            answer_parts.append(block.text)
        elif isinstance(block, dict) and "text" in block:
            answer_parts.append(block["text"])

    raw_answer = "".join(answer_parts).strip()
    sources = get_unique_sources(chunks)

    # If model responded that it could not find info, clear sources
    lowered = raw_answer.lower()
    if (
        "no relevant information found" in lowered
        or "could not find information" in lowered
    ):
        sources = []

    return raw_answer, sources
