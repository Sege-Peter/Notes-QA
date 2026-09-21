"""FastAPI web server and interactive live UI for notes-qa."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from notes_qa.config import (
    COLLECTION_NAME,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_DB_PATH,
    get_anthropic_api_key,
    get_db_path,
)
from notes_qa.generate import generate_answer
from notes_qa.ingest import (
    DocumentChunk,
    get_chroma_client,
    get_embedding_function,
    ingest_folder,
)
from notes_qa.retrieve import retrieve_chunks


class IngestRequest(BaseModel):
    folder: str
    rebuild: bool = False


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    api_key: Optional[str] = None
    model: Optional[str] = None
    enable_hybrid: bool = True


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    enable_hybrid: bool = True


def create_app() -> FastAPI:
    app = FastAPI(
        title="notes-qa Live UI",
        description="Interactive RAG interface for personal notes and PDFs",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        """Get vector database and configuration status."""
        db_path = get_db_path()
        client = get_chroma_client(db_path)
        try:
            collection = client.get_collection(COLLECTION_NAME)
            count = collection.count()
        except Exception:
            count = 0

        api_key = get_anthropic_api_key()
        return {
            "status": "online",
            "db_path": db_path,
            "collection_name": COLLECTION_NAME,
            "indexed_chunks": count,
            "has_anthropic_key": bool(api_key),
            "default_model": DEFAULT_ANTHROPIC_MODEL,
        }

    @app.post("/api/ingest")
    async def ingest_endpoint(req: IngestRequest) -> dict[str, Any]:
        """Ingest documents from a local folder."""
        folder_path = Path(req.folder).resolve()
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail=f"Folder '{req.folder}' does not exist.")
        if not folder_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Path '{req.folder}' is not a directory.")

        try:
            stats = ingest_folder(folder=folder_path, rebuild=req.rebuild)
            return {"success": True, "stats": stats}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/upload")
    async def upload_endpoint(
        files: list[UploadFile] = File(...),
        rebuild: bool = False,
    ) -> dict[str, Any]:
        """Upload and ingest PDF or Markdown files directly."""
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        upload_dir = Path("./uploaded_notes").resolve()
        upload_dir.mkdir(exist_ok=True)

        saved_files = []
        for file in files:
            filename = Path(file.filename).name
            dest = upload_dir / filename
            with dest.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(str(dest))

        stats = ingest_folder(folder=upload_dir, rebuild=rebuild)
        return {"success": True, "saved_files": saved_files, "stats": stats}

    @app.post("/api/retrieve")
    async def search_endpoint(req: SearchRequest) -> dict[str, Any]:
        """Retrieve relevant chunks with similarity and hybrid BM25 scores."""
        chunks = retrieve_chunks(
            query=req.query,
            top_k=req.top_k,
            enable_hybrid=req.enable_hybrid,
        )
        return {
            "query": req.query,
            "count": len(chunks),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "file_name": c.file_name,
                    "doc_type": c.doc_type,
                    "page": c.page,
                    "heading": c.heading,
                    "source_citation": c.source_citation,
                    "inline_citation_tag": c.inline_citation_tag,
                    "score": round(c.score, 3) if c.score is not None else None,
                    "distance": round(c.distance, 3) if c.distance is not None else None,
                }
                for c in chunks
            ],
        }

    @app.post("/api/ask")
    async def ask_endpoint(req: AskRequest) -> dict[str, Any]:
        """Retrieve context and generate grounded answer using Claude."""
        chunks = retrieve_chunks(
            query=req.question,
            top_k=req.top_k,
            enable_hybrid=req.enable_hybrid,
        )

        if not chunks:
            return {
                "answer": "No relevant information found in your notes for this question.",
                "sources": [],
                "chunks": [],
            }

        try:
            answer, sources = generate_answer(
                query=req.question,
                chunks=chunks,
                api_key=req.api_key,
                model=req.model,
            )
        except ValueError as val_err:
            raise HTTPException(status_code=400, detail=str(val_err))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

        return {
            "answer": answer,
            "sources": sources,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "file_name": c.file_name,
                    "doc_type": c.doc_type,
                    "page": c.page,
                    "heading": c.heading,
                    "source_citation": c.source_citation,
                    "inline_citation_tag": c.inline_citation_tag,
                    "score": round(c.score, 3) if c.score is not None else None,
                    "distance": round(c.distance, 3) if c.distance is not None else None,
                }
                for c in chunks
            ],
        }

    @app.get("/", response_class=HTMLResponse)
    async def serve_ui() -> str:
        """Serve the live single-page web UI."""
        return HTML_PAGE

    return app


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>notes-qa | Grounded Notes Q&A</title>
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Marked.js for Markdown -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              50: '#eef2ff',
              100: '#e0e7ff',
              500: '#6366f1',
              600: '#4f46e5',
              700: '#4338ca',
            }
          }
        }
      }
    }
  </script>
  <style>
    .citation-badge {
      cursor: pointer;
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.3);
      padding: 2px 6px;
      border-radius: 4px;
      font-family: monospace;
      font-size: 0.85em;
      transition: all 0.2s ease;
    }
    .citation-badge:hover {
      background: rgba(99, 102, 241, 0.3);
      border-color: #6366f1;
    }
    .markdown-body pre {
      background: #0f172a;
      color: #f8fafc;
      padding: 12px;
      border-radius: 8px;
      overflow-x: auto;
      margin: 10px 0;
    }
    .markdown-body code {
      font-family: monospace;
      background: rgba(148, 163, 184, 0.2);
      padding: 2px 5px;
      border-radius: 4px;
    }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans antialiased selection:bg-brand-500 selection:text-white">

  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-lg bg-gradient-to-tr from-brand-600 to-indigo-400 flex items-center justify-center font-black text-white shadow-lg shadow-brand-500/20">
          N
        </div>
        <div>
          <span class="font-bold text-lg tracking-tight text-white flex items-center gap-2">
            notes-qa
            <span class="text-xs font-semibold px-2 py-0.5 rounded-full bg-brand-500/10 text-brand-400 border border-brand-500/20">v0.1.0</span>
          </span>
          <p class="text-xs text-slate-400">RAG with Hybrid Search & Grounded Citations</p>
        </div>
      </div>

      <div class="flex items-center gap-3">
        <div id="statusBadge" class="flex items-center gap-2 px-3 py-1 rounded-full text-xs bg-slate-800 border border-slate-700 text-slate-300">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span id="statusText">Checking status...</span>
        </div>
        <button onclick="toggleSettingsModal()" class="px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 transition flex items-center gap-1.5">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
          Settings
        </button>
      </div>
    </div>
  </header>

  <!-- Main Grid -->
  <main class="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">

    <!-- Left Column: Ingestion & Document Manager -->
    <aside class="lg:col-span-4 space-y-6">
      
      <!-- Ingest Card -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
        <h2 class="text-sm font-semibold text-slate-200 flex items-center justify-between">
          <span>Index Documents</span>
          <span class="text-xs text-brand-400 font-normal">PDF + Markdown</span>
        </h2>
        
        <!-- Folder Ingestion Form -->
        <div class="space-y-3">
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Folder Path</label>
            <input type="text" id="folderInput" value="./sample_notes" placeholder="./my-notes" class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-brand-500 font-mono">
          </div>

          <div class="flex items-center gap-2">
            <input type="checkbox" id="rebuildCheck" class="rounded bg-slate-950 border-slate-800 text-brand-600 focus:ring-0">
            <label for="rebuildCheck" class="text-xs text-slate-400 cursor-pointer">Rebuild index from scratch</label>
          </div>

          <button id="ingestBtn" onclick="triggerIngest()" class="w-full py-2 bg-brand-600 hover:bg-brand-500 text-white text-sm font-medium rounded-lg shadow-sm shadow-brand-500/20 transition flex items-center justify-center gap-2">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
            Ingest Folder
          </button>
        </div>

        <!-- File Upload Section -->
        <div class="border-t border-slate-800 pt-4">
          <label class="block text-xs font-medium text-slate-400 mb-2">Or Upload Files (.pdf, .md)</label>
          <div class="border-2 border-dashed border-slate-800 hover:border-brand-500/50 rounded-lg p-4 text-center cursor-pointer transition bg-slate-950/50" onclick="document.getElementById('fileUploadInput').click()">
            <input type="file" id="fileUploadInput" multiple accept=".pdf,.md,.markdown" class="hidden" onchange="handleFileUpload(event)">
            <svg class="w-6 h-6 mx-auto text-slate-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
            <p class="text-xs text-slate-300 font-medium">Click to select PDF or Markdown files</p>
            <p class="text-[10px] text-slate-500 mt-1">Files will be placed in ./uploaded_notes and indexed</p>
          </div>
        </div>

        <!-- Ingest Log Output -->
        <div id="ingestLog" class="hidden text-xs bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono space-y-1 text-slate-300">
        </div>
      </div>

      <!-- Quick Tips Card -->
      <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 text-xs text-slate-400 space-y-3">
        <h3 class="font-semibold text-slate-300 flex items-center gap-1.5">
          <svg class="w-4 h-4 text-brand-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
          How It Works
        </h3>
        <p>• <strong>Hybrid Search</strong> combines 70% Chroma vector cosine similarity + 30% BM25 keyword matching for exact acronyms, code identifiers, and phrases.</p>
        <p>• <strong>Grounded Citations</strong> link every claim back to <span class="font-mono text-brand-300">[file, page/section]</span>.</p>
        <p>• <strong>Hallucination Defense</strong> automatically blocks answers when no chunks exceed the relevance cutoff.</p>
      </div>

    </aside>

    <!-- Right Column: Interactive Q&A and Citations -->
    <section class="lg:col-span-8 flex flex-col space-y-6">

      <!-- Query Box -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm space-y-3">
        <div class="flex items-center gap-2">
          <input type="text" id="queryInput" placeholder="Ask a question over your notes (e.g. 'What did I write about rate limiting?')" class="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-brand-500 font-normal transition" onkeydown="if(event.key==='Enter') executeAsk()">
          <button id="askBtn" onclick="executeAsk()" class="px-5 py-3 bg-brand-600 hover:bg-brand-500 text-white font-medium text-sm rounded-lg shadow-sm shadow-brand-500/20 transition flex items-center gap-2">
            <span>Ask</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
          </button>
        </div>

        <!-- Sample Prompts -->
        <div class="flex flex-wrap items-center gap-2 text-xs">
          <span class="text-slate-500">Try:</span>
          <button onclick="setQuery('What did I write about rate limiting?')" class="px-2.5 py-1 bg-slate-800/80 hover:bg-slate-700 text-slate-300 rounded-full border border-slate-700 transition">Rate Limiting</button>
          <button onclick="setQuery('Explain Write-Ahead Logging in databases')" class="px-2.5 py-1 bg-slate-800/80 hover:bg-slate-700 text-slate-300 rounded-full border border-slate-700 transition">Write-Ahead Log (WAL)</button>
          <button onclick="setQuery('What does Slow Start do in TCP?')" class="px-2.5 py-1 bg-slate-800/80 hover:bg-slate-700 text-slate-300 rounded-full border border-slate-700 transition">TCP Slow Start</button>
          <button onclick="setQuery('Who won the 1998 World Cup?')" class="px-2.5 py-1 bg-slate-800/80 hover:bg-slate-700 text-rose-400/80 rounded-full border border-rose-900/40 transition">Irrelevant Test</button>
        </div>
      </div>

      <!-- Loading State -->
      <div id="loadingIndicator" class="hidden bg-slate-900 border border-slate-800 rounded-xl p-8 text-center space-y-3">
        <div class="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
        <p class="text-sm text-slate-300 font-medium">Retrieving relevant notes & synthesizing grounded answer...</p>
        <p class="text-xs text-slate-500">Evaluating vector similarity and BM25 rank fusion</p>
      </div>

      <!-- Results Display Container -->
      <div id="resultsContainer" class="hidden space-y-6">

        <!-- Answer Card -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm space-y-4">
          <div class="flex items-center justify-between border-b border-slate-800 pb-3">
            <h2 class="text-base font-semibold text-white flex items-center gap-2">
              <svg class="w-5 h-5 text-brand-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
              Grounded Answer
            </h2>
            <button onclick="copyAnswer()" class="text-xs text-slate-400 hover:text-white flex items-center gap-1">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"></path></svg>
              Copy
            </button>
          </div>

          <div id="answerContent" class="markdown-body text-slate-200 text-sm leading-relaxed space-y-3">
          </div>

          <!-- Sources Badges Summary -->
          <div id="sourcesSummary" class="border-t border-slate-800 pt-3">
            <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Sources Referenced:</p>
            <div id="sourcesList" class="flex flex-wrap gap-2"></div>
          </div>
        </div>

        <!-- Retrieved Context Chunks / Receipts -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm space-y-4">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
              Retrieved Context Receipts (<span id="retrievedCount">0</span> chunks)
            </h3>
            <span class="text-xs text-slate-500">Ranked by Hybrid Score</span>
          </div>

          <div id="chunksList" class="space-y-3">
          </div>
        </div>

      </div>

    </section>

  </main>

  <!-- Settings Modal -->
  <div id="settingsModal" class="hidden fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="bg-slate-900 border border-slate-800 rounded-xl max-w-md w-full p-6 space-y-5 shadow-2xl">
      <div class="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 class="font-bold text-white text-base">Settings & API Keys</h3>
        <button onclick="toggleSettingsModal()" class="text-slate-400 hover:text-white">&times;</button>
      </div>

      <div class="space-y-4 text-sm">
        <div>
          <label class="block text-xs font-medium text-slate-400 mb-1">Anthropic API Key</label>
          <input type="password" id="apiKeyInput" placeholder="sk-ant-api..." class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-brand-500 font-mono">
          <p class="text-[11px] text-slate-500 mt-1">If set, overrides the environment variable ANTHROPIC_API_KEY for this session.</p>
        </div>

        <div>
          <label class="block text-xs font-medium text-slate-400 mb-1">Claude Model</label>
          <select id="modelSelect" class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-brand-500">
            <option value="claude-sonnet-4-6" selected>claude-sonnet-4-6 (Default)</option>
            <option value="claude-3-5-sonnet-20241022">claude-3-5-sonnet-20241022</option>
            <option value="claude-3-haiku-20240307">claude-3-haiku-20240307</option>
          </select>
        </div>

        <div>
          <label class="block text-xs font-medium text-slate-400 mb-1">Top-K Retrieval</label>
          <div class="flex items-center gap-3">
            <input type="range" id="topKRange" min="1" max="10" value="5" class="flex-1 accent-brand-500" oninput="document.getElementById('topKVal').innerText = this.value">
            <span id="topKVal" class="text-xs font-mono font-bold w-6 text-right">5</span>
          </div>
        </div>

        <div class="flex items-center justify-between pt-2">
          <label class="text-xs font-medium text-slate-300">Hybrid Search (Vector + BM25)</label>
          <input type="checkbox" id="hybridToggle" checked class="rounded bg-slate-950 border-slate-800 text-brand-600 focus:ring-0">
        </div>
      </div>

      <div class="border-t border-slate-800 pt-4 flex justify-end gap-2">
        <button onclick="saveSettings()" class="px-4 py-2 bg-brand-600 hover:bg-brand-500 text-white text-xs font-medium rounded-lg transition">Save Settings</button>
      </div>
    </div>
  </div>

  <script>
    let userApiKey = localStorage.getItem('notes_qa_api_key') || '';
    let selectedModel = localStorage.getItem('notes_qa_model') || 'claude-sonnet-4-6';

    document.getElementById('apiKeyInput').value = userApiKey;
    document.getElementById('modelSelect').value = selectedModel;

    function toggleSettingsModal() {
      const modal = document.getElementById('settingsModal');
      modal.classList.toggle('hidden');
    }

    function saveSettings() {
      userApiKey = document.getElementById('apiKeyInput').value.trim();
      selectedModel = document.getElementById('modelSelect').value;
      localStorage.setItem('notes_qa_api_key', userApiKey);
      localStorage.setItem('notes_qa_model', selectedModel);
      toggleSettingsModal();
      refreshStatus();
    }

    function setQuery(text) {
      document.getElementById('queryInput').value = text;
      executeAsk();
    }

    async function refreshStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const badge = document.getElementById('statusBadge');
        const text = document.getElementById('statusText');
        
        text.innerHTML = `DB: <span class="font-bold text-emerald-400">${data.indexed_chunks}</span> segments indexed`;
        if (!data.has_anthropic_key && !userApiKey) {
          text.innerHTML += ` <span class="text-amber-400 ml-1">(No API Key)</span>`;
        }
      } catch (err) {
        document.getElementById('statusText').innerText = 'Backend Offline';
      }
    }

    async function triggerIngest() {
      const folder = document.getElementById('folderInput').value.trim();
      const rebuild = document.getElementById('rebuildCheck').checked;
      const logDiv = document.getElementById('ingestLog');
      const btn = document.getElementById('ingestBtn');

      if (!folder) return alert('Please provide a folder path');

      btn.disabled = true;
      btn.innerHTML = '<div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div> Scanning & Indexing...';
      logDiv.classList.remove('hidden');
      logDiv.innerHTML = `<span class="text-slate-500">Scanning ${folder} ...</span><br>`;

      try {
        const res = await fetch('/api/ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ folder, rebuild })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Ingestion failed');

        const s = data.stats;
        logDiv.innerHTML = `
          <div class="text-emerald-400 font-bold mb-1">Ingestion Complete!</div>
          <div>Files scanned: ${s.total_files} (${s.pdf_count} PDFs, ${s.md_count} Markdown)</div>
          <div>Chunks created: ${s.chunk_count} segments</div>
          <div>Time elapsed: ${s.elapsed_seconds.toFixed(2)}s</div>
          <div class="text-slate-500">Stored at: ${s.db_path}</div>
        `;
        refreshStatus();
      } catch (err) {
        logDiv.innerHTML = `<span class="text-rose-400 font-bold">Error:</span> ${err.message}`;
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg> Ingest Folder';
      }
    }

    async function handleFileUpload(event) {
      const files = event.target.files;
      if (!files || files.length === 0) return;

      const formData = new FormData();
      for (const file of files) {
        formData.append('files', file);
      }

      const logDiv = document.getElementById('ingestLog');
      logDiv.classList.remove('hidden');
      logDiv.innerHTML = `<span class="text-slate-500">Uploading and indexing ${files.length} file(s)...</span>`;

      try {
        const res = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Upload failed');

        logDiv.innerHTML = `
          <div class="text-emerald-400 font-bold mb-1">Files Uploaded & Indexed!</div>
          <div>Chunks added: ${data.stats.chunk_count} segments</div>
        `;
        refreshStatus();
      } catch (err) {
        logDiv.innerHTML = `<span class="text-rose-400 font-bold">Error:</span> ${err.message}`;
      }
    }

    async function executeAsk() {
      const question = document.getElementById('queryInput').value.trim();
      if (!question) return;

      const top_k = parseInt(document.getElementById('topKRange').value, 10) || 5;
      const enable_hybrid = document.getElementById('hybridToggle').checked;

      const loading = document.getElementById('loadingIndicator');
      const results = document.getElementById('resultsContainer');
      const askBtn = document.getElementById('askBtn');

      loading.classList.remove('hidden');
      results.classList.add('hidden');
      askBtn.disabled = true;

      try {
        const res = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question,
            top_k,
            api_key: userApiKey || null,
            model: selectedModel || 'claude-sonnet-4-6',
            enable_hybrid
          })
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to generate answer');

        // Render Markdown Answer with highlighted inline citations
        let parsed = marked.parse(data.answer);
        parsed = parsed.replace(/\\[([^\\]]+)\\]/g, '<span class="citation-badge">[$1]</span>');
        document.getElementById('answerContent').innerHTML = parsed;

        // Render Sources
        const sourcesDiv = document.getElementById('sourcesList');
        sourcesDiv.innerHTML = '';
        if (data.sources && data.sources.length > 0) {
          data.sources.forEach(src => {
            const badge = document.createElement('span');
            badge.className = 'px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-xs font-mono text-brand-300';
            badge.innerText = src;
            sourcesDiv.appendChild(badge);
          });
          document.getElementById('sourcesSummary').classList.remove('hidden');
        } else {
          document.getElementById('sourcesSummary').classList.add('hidden');
        }

        // Render Context Receipts
        document.getElementById('retrievedCount').innerText = data.chunks ? data.chunks.length : 0;
        const chunksList = document.getElementById('chunksList');
        chunksList.innerHTML = '';

        if (data.chunks && data.chunks.length > 0) {
          data.chunks.forEach((chunk, i) => {
            const card = document.createElement('div');
            card.className = 'p-4 rounded-lg bg-slate-950 border border-slate-800 text-xs space-y-2';
            const icon = chunk.doc_type === 'pdf' ? '📄 PDF' : '📝 MD';
            card.innerHTML = `
              <div class="flex items-center justify-between">
                <span class="font-mono font-semibold text-slate-300">${icon} ${chunk.source_citation}</span>
                <div class="flex items-center gap-2">
                  <span class="px-2 py-0.5 rounded bg-brand-500/10 text-brand-300 border border-brand-500/20 font-mono">Hybrid Score: ${chunk.score !== null ? chunk.score : 'N/A'}</span>
                  <span class="px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">Dist: ${chunk.distance !== null ? chunk.distance : 'N/A'}</span>
                </div>
              </div>
              <p class="text-slate-300 leading-relaxed font-mono bg-slate-900/60 p-2.5 rounded border border-slate-800/80 whitespace-pre-wrap">${chunk.text}</p>
            `;
            chunksList.appendChild(card);
          });
        } else {
          chunksList.innerHTML = '<p class="text-xs text-slate-500">No context chunks matched this query.</p>';
        }

        results.classList.remove('hidden');
      } catch (err) {
        alert(err.message);
      } finally {
        loading.classList.add('hidden');
        askBtn.disabled = false;
      }
    }

    function copyAnswer() {
      const text = document.getElementById('answerContent').innerText;
      navigator.clipboard.writeText(text);
      alert('Answer copied to clipboard!');
    }

    // Initial load
    refreshStatus();
  </script>
</body>
</html>
"""
