"""Configuration settings for notes-qa."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load .env if present
load_dotenv()

DEFAULT_DB_PATH = os.getenv("NOTES_QA_DB_PATH", "./.db")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "local")
DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
COLLECTION_NAME = "notes_qa"

# Cosine distance cutoff: distance > 0.85 indicates low relevance
DEFAULT_SIMILARITY_DISTANCE_THRESHOLD = float(
    os.getenv("SIMILARITY_DISTANCE_THRESHOLD", "0.85")
)


def get_anthropic_api_key() -> str | None:
    """Return the Anthropic API key from environment, or None if not set."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if key and key.strip() and key.strip() != "your_anthropic_api_key_here":
        return key.strip()
    return None


def get_gemini_api_key() -> str | None:
    """Return the Google Gemini API key from environment, or None if not set."""
    key = os.getenv("GEMINI_API_KEY")
    if key and key.strip() and key.strip() != "your_gemini_api_key_here":
        return key.strip()
    return None


def get_db_path(custom_path: str | None = None) -> str:
    """Resolve database path with priority given to custom arguments."""
    return custom_path if custom_path is not None else DEFAULT_DB_PATH


def get_embedding_model() -> str:
    """Return configured embedding model name ('local', 'openai', 'anthropic')."""
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).lower()

