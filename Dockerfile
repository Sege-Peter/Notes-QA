# notes-qa web UI as a public demo (Hugging Face Spaces, Docker SDK, port 7860).
FROM python:3.11-slim

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    NOTES_QA_PUBLIC=1 \
    LLM_PROVIDER=offline \
    EMBEDDING_MODEL=local \
    NOTES_QA_DB_PATH=/home/user/app/.db \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build
RUN pip install --no-cache-dir --user --extra-index-url https://download.pytorch.org/whl/cpu ".[ui]"

# download the embedding model at build time so the first request is fast
RUN python -c "from notes_qa.ingest import get_embedding_function; get_embedding_function()(['warm up'])"

COPY --chown=user sample_notes ./sample_notes

EXPOSE 7860
CMD ["uvicorn", "notes_qa.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
