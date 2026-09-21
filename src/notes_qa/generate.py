"""Prompt construction, LLM generation, and citation formatting supporting Claude, Gemini, and Zero API Key (Offline) mode."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, List, Optional, Tuple

from notes_qa.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GEMINI_MODEL,
    get_anthropic_api_key,
    get_gemini_api_key,
)
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


def _try_ollama_generation(
    query: str,
    chunks: list[DocumentChunk],
    model: str = "llama3",
) -> Optional[str]:
    """Attempt generation via local Ollama instance if running."""
    try:
        context_text = build_context_block(chunks)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Provide your grounded answer with inline citations:"
        )
        data = json.dumps(
            {"model": model, "prompt": prompt, "stream": False}
        ).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response")
    except Exception:
        pass
    return None


def generate_answer_anthropic(
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


def generate_answer_gemini(
    query: str,
    chunks: list[DocumentChunk],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    client: Any = None,
) -> tuple[str, list[str]]:
    """Generate grounded answer from Google Gemini with inline citations and source list."""
    if not chunks:
        return ("No relevant information found in your notes for this question.", [])

    resolved_key = api_key or get_gemini_api_key()
    if not resolved_key and client is None:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add it to your .env file or environment, or select Zero API Key mode."
        )

    if client is None:
        from google import genai

        client = genai.Client(api_key=resolved_key)

    chosen_model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    context_text = build_context_block(chunks)

    user_message = (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Provide your grounded answer with inline citations:"
    )

    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=1024,
    )

    response = client.models.generate_content(
        model=chosen_model,
        contents=user_message,
        config=config,
    )

    raw_answer = (response.text or "").strip()
    sources = get_unique_sources(chunks)

    lowered = raw_answer.lower()
    if (
        "no relevant information found" in lowered
        or "could not find information" in lowered
        or "not enough information" in lowered
    ):
        sources = []

    return raw_answer, sources


def generate_answer_offline(
    query: str,
    chunks: list[DocumentChunk],
    model: Optional[str] = None,
) -> tuple[str, list[str]]:
    """Generate grounded answer in Zero API Key mode (extractive synthesizer or local Ollama)."""
    if not chunks:
        return ("No relevant information found in your notes for this question.", [])

    # Check for local Ollama if explicitly requested or configured
    ollama_model = None
    if model and model.startswith("ollama"):
        ollama_model = model.replace("ollama:", "") if ":" in model else "llama3"
    elif os.getenv("OLLAMA_MODEL"):
        ollama_model = os.getenv("OLLAMA_MODEL")

    if ollama_model:
        ollama_answer = _try_ollama_generation(query, chunks, model=ollama_model)
        if ollama_answer:
            sources = get_unique_sources(chunks)
            lowered = ollama_answer.lower()
            if "no relevant information found" in lowered or "could not find" in lowered:
                sources = []
            return ollama_answer.strip(), sources

    # Extractive Grounded Synthesizer (pure Python, 100% offline, zero external API keys)
    stopwords = {
        "what", "did", "i", "write", "about", "the", "a", "an", "is", "are",
        "was", "were", "for", "to", "in", "of", "and", "how", "do", "does",
        "can", "tell", "me", "explain", "why", "when", "which", "where",
        "my", "notes", "say", "does", "any", "some", "with", "from", "on"
    }
    query_words = set(
        w.lower()
        for w in re.findall(r"\b[a-zA-Z0-9_\-]+\b", query)
        if w.lower() not in stopwords and len(w) > 1
    )

    scored_sentences: list[tuple[float, str, DocumentChunk]] = []
    seen_texts: set[str] = set()

    for chunk in chunks:
        raw_sentences = re.split(r"(?<=[.!?\n])\s+", chunk.text)
        for s in raw_sentences:
            s_clean = s.strip().lstrip("-*• ")
            if len(s_clean) < 15 or s_clean in seen_texts:
                continue

            words = set(w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]+\b", s_clean))
            overlap = len(words & query_words)
            if overlap > 0 or not query_words:
                score = (overlap * 2.0) + (chunk.score or 0.0)
                scored_sentences.append((score, s_clean, chunk))
                seen_texts.add(s_clean)

    if not scored_sentences:
        return ("No relevant information found in your notes for this question.", [])

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_sentences[:4]

    lines = ["Based on your notes:"]
    chunks_used: list[DocumentChunk] = []

    for _, sentence, chunk in top_matches:
        citation = chunk.inline_citation_tag
        lines.append(f"- {sentence} {citation}")
        chunks_used.append(chunk)

    answer = "\n".join(lines)
    sources = get_unique_sources(chunks_used)
    return answer, sources


def generate_answer(
    query: str,
    chunks: list[DocumentChunk],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    client: Any = None,
    provider: Optional[str] = None,
) -> tuple[str, list[str]]:
    """Generate grounded answer routing to Claude, Gemini, or Zero API Key offline mode."""
    if not chunks:
        return ("No relevant information found in your notes for this question.", [])

    prov = (provider or os.getenv("LLM_PROVIDER", "")).strip().lower()

    if not prov or prov == "auto":
        if model and "gemini" in model.lower():
            prov = "gemini"
        elif model in ("offline", "zero-key", "local"):
            prov = "offline"
        elif api_key and api_key.startswith("AIza"):
            prov = "gemini"
        elif get_gemini_api_key() and not get_anthropic_api_key():
            prov = "gemini"
        else:
            prov = "anthropic"

    if prov == "gemini":
        return generate_answer_gemini(
            query=query,
            chunks=chunks,
            api_key=api_key,
            model=model,
            client=client,
        )
    elif prov in ("offline", "zero-key", "local"):
        return generate_answer_offline(
            query=query,
            chunks=chunks,
            model=model,
        )
    else:
        return generate_answer_anthropic(
            query=query,
            chunks=chunks,
            api_key=api_key,
            model=model,
            client=client,
        )
