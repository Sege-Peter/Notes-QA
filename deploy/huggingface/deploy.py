"""Publish (or update) the notes-qa live demo on Hugging Face Spaces.

Usage:
    hf auth login                       # once, with a token that has write access
    python deploy/huggingface/deploy.py [space-name]   # default space name: notes-qa

Only the files the demo needs are uploaded; .env, the local index and uploads never leave your machine.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
SPACE_FILES = ["Dockerfile", ".dockerignore", "pyproject.toml", "src/**", "sample_notes/**"]


def main() -> None:
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{sys.argv[1] if len(sys.argv) > 1 else 'notes-qa'}"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=ROOT,
        allow_patterns=SPACE_FILES,
        ignore_patterns=["**/__pycache__/**", "**/*.egg-info/**", "**/*.pyc"],
        commit_message="Deploy notes-qa demo",
    )
    api.upload_file(
        path_or_fileobj=str(ROOT / "deploy" / "huggingface" / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Update Space card",
    )
    print(f"Space: https://huggingface.co/spaces/{repo_id}")
    print(f"App:   https://{repo_id.replace('/', '-').replace('_', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
