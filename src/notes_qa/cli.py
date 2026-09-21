"""Command-line interface for notes-qa."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from notes_qa.config import DEFAULT_DB_PATH
from notes_qa.generate import format_qa_output, generate_answer
from notes_qa.ingest import ingest_folder
from notes_qa.retrieve import retrieve_chunks


@click.group()
@click.version_option(version="0.1.0", prog_name="notes-qa")
def main() -> None:
    """notes-qa: Grounded question answering over your personal notes and PDFs."""
    pass


@main.command("ingest")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--rebuild",
    is_flag=True,
    default=False,
    help="Rebuild the vector store from scratch.",
)
@click.option(
    "--db-path",
    default=None,
    help=f"Path where vector store is persisted (default: {DEFAULT_DB_PATH}).",
)
def ingest_cmd(folder: str, rebuild: bool, db_path: str | None) -> None:
    """Index a folder containing Markdown notes and PDFs."""
    folder_path = Path(folder)
    click.echo(f"Scanning {folder} ...")

    try:
        stats = ingest_folder(
            folder=folder_path,
            db_path=db_path,
            rebuild=rebuild,
        )
    except Exception as err:
        click.secho(f"Error during ingestion: {err}", fg="red", err=True)
        sys.exit(1)

    if stats["total_files"] == 0:
        click.secho(
            f"No supported documents found in {folder} (searched for .pdf, .md, .markdown).",
            fg="yellow",
        )
        if stats.get("skipped_count", 0) > 0:
            click.echo(f"Skipped {stats['skipped_count']} unsupported files.")
        return

    click.echo(
        f"Found {stats['total_files']} files "
        f"({stats['pdf_count']} PDFs, {stats['md_count']} Markdown)"
    )
    if stats.get("skipped_count", 0) > 0:
        click.echo(f"Skipped {stats['skipped_count']} unsupported files.")
    click.echo(f"Chunked into {stats['chunk_count']} segments")
    click.echo(f"Embedding... done in {stats['elapsed_seconds']:.1f}s")
    click.echo(f"Stored in {stats['db_path']}")


@main.command("ask")
@click.argument("question", type=str)
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    type=int,
    help="Number of document chunks to retrieve.",
)
@click.option(
    "--db-path",
    default=None,
    help=f"Path where vector store is persisted (default: {DEFAULT_DB_PATH}).",
)
def ask_cmd(question: str, top_k: int, db_path: str | None) -> None:
    """Ask a question grounded in your indexed notes and PDFs."""
    try:
        chunks = retrieve_chunks(
            query=question,
            top_k=top_k,
            db_path=db_path,
        )
    except Exception as err:
        click.secho(f"Error retrieving context: {err}", fg="red", err=True)
        sys.exit(1)

    if not chunks:
        click.echo("Answer:")
        click.echo("No relevant information found in your notes for this question.")
        return

    try:
        answer, sources = generate_answer(query=question, chunks=chunks)
    except ValueError as val_err:
        click.secho(f"Configuration error: {val_err}", fg="yellow", err=True)
        sys.exit(1)
    except Exception as err:
        click.secho(f"Error generating answer: {err}", fg="red", err=True)
        sys.exit(1)

    output = format_qa_output(answer, sources)
    click.echo(output)


@main.command("ui")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind server to.")
@click.option("--port", default=8000, show_default=True, type=int, help="Port to bind server to.")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
def ui_cmd(host: str, port: int, reload: bool) -> None:
    """Launch the interactive Web UI."""
    import uvicorn

    click.echo(f"Starting notes-qa Live UI at http://{host}:{port} ...")
    uvicorn.run("notes_qa.server:create_app", host=host, port=port, factory=True, reload=reload)


if __name__ == "__main__":
    main()
