// notes-qa in the browser: a port of the Python pipeline (ingest -> embed -> hybrid
// retrieve -> grounded answer) that answers the web UI's /api/* calls locally.
// Nothing is uploaded: documents, the index and API keys stay in this browser.

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
import * as pdfjsLib from './vendor/pdf.min.mjs';

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL('./vendor/pdf.worker.min.mjs', import.meta.url).href;
env.allowLocalModels = false;

const COLLECTION_NAME = 'notes_qa';
const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2'; // all-MiniLM-L6-v2, as in the Python build
const DISTANCE_THRESHOLD = 0.85;
const NO_INFO = 'No relevant information found in your notes for this question.';
const DEFAULT_GEMINI_MODEL = 'gemini-2.5-flash';
const DEFAULT_ANTHROPIC_MODEL = 'claude-sonnet-4-6';
const SYSTEM_PROMPT = `You are a helpful assistant that answers questions based strictly on the user's personal notes and documents.

Rules:
1. Answer ONLY based on the facts provided in the Context below. Do NOT assume, extrapolate, or use outside knowledge.
2. If the provided context does not contain enough information to answer the question, state:
   "No relevant information found in your notes for this question."
3. Cite your sources inline immediately following relevant claims, using the citation tags indicated for each excerpt (for example: [notes.md, "Section Name"] or [document.pdf, p. 12]).
4. Keep the answer clear, grounded, and concise.`;

// ---------------------------------------------------------------- storage

let index = []; // { chunk_id, text, file_name, doc_type, page, heading, embedding }

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('notes-qa', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('index');
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function loadIndex() {
  try {
    const db = await openDb();
    index = await new Promise((resolve) => {
      const r = db.transaction('index').objectStore('index').get(COLLECTION_NAME);
      r.onsuccess = () => resolve(r.result || []);
      r.onerror = () => resolve([]);
    });
  } catch {
    index = [];
  }
}
async function saveIndex() {
  try {
    const db = await openDb();
    await new Promise((resolve) => {
      const tx = db.transaction('index', 'readwrite');
      tx.objectStore('index').put(index, COLLECTION_NAME);
      tx.oncomplete = resolve;
      tx.onerror = resolve;
    });
  } catch {
    /* private mode etc.: keep the in-memory index */
  }
}

// ---------------------------------------------------------------- citations (DocumentChunk)

const sourceCitation = (c) =>
  c.doc_type === 'pdf' && c.page != null ? `${c.file_name} — page ${c.page}`
    : c.doc_type === 'markdown' && c.heading ? `${c.file_name} — "${c.heading}"` : c.file_name;
const inlineCitationTag = (c) =>
  c.doc_type === 'pdf' && c.page != null ? `[${c.file_name}, p. ${c.page}]`
    : c.doc_type === 'markdown' && c.heading ? `[${c.file_name}, "${c.heading}"]` : `[${c.file_name}]`;

// ---------------------------------------------------------------- chunking (ingest.py)

function splitTextIntoChunks(text, chunkSizeTokens = 500, overlapTokens = 50) {
  const chunkSizeWords = Math.max(50, Math.floor(chunkSizeTokens * 0.75));
  const overlapWords = Math.max(5, Math.floor(overlapTokens * 0.75));
  const words = text.split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  if (words.length <= chunkSizeWords) return [words.join(' ')];
  const chunks = [];
  const step = Math.max(1, chunkSizeWords - overlapWords);
  let start = 0;
  while (start < words.length) {
    chunks.push(words.slice(start, start + chunkSizeWords).join(' '));
    start += step;
    if (start + overlapWords >= words.length) break;
  }
  return chunks;
}

function parseMarkdown(fileName, content) {
  const chunks = [];
  let heading = null;
  let lines = [];
  let sectionIndex = 0;
  const flush = () => {
    const text = lines.join('\n').trim();
    if (!text) return;
    splitTextIntoChunks(text).forEach((segment, sub) => {
      chunks.push({ chunk_id: `${fileName}_s${sectionIndex}_c${sub}`, text: segment, file_name: fileName, doc_type: 'markdown', page: null, heading });
    });
    sectionIndex += 1;
  };
  for (const line of content.split(/\r?\n/)) {
    const m = line.match(/^(#{1,6})\s+(.*)$/);
    if (m) {
      if (lines.length) { flush(); lines = []; }
      heading = m[2].trim();
    } else {
      lines.push(line);
    }
  }
  if (lines.length) flush();
  return chunks;
}

async function parsePdf(fileName, bytes) {
  const chunks = [];
  const pdf = await pdfjsLib.getDocument({ data: bytes }).promise;
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const content = await page.getTextContent();
    const text = content.items.map((it) => it.str + (it.hasEOL ? '\n' : ' ')).join('').trim();
    if (!text) continue;
    splitTextIntoChunks(text).forEach((segment, seg) => {
      chunks.push({ chunk_id: `${fileName}_p${p}_c${seg}`, text: segment, file_name: fileName, doc_type: 'pdf', page: p, heading: null });
    });
  }
  return chunks;
}

// ---------------------------------------------------------------- embeddings

let extractorPromise = null;
function getExtractor() {
  if (!extractorPromise) {
    setStatus('Loading the embedding model (first visit only, ~25 MB)…');
    extractorPromise = pipeline('feature-extraction', EMBED_MODEL);
  }
  return extractorPromise;
}
async function embed(texts) {
  const extractor = await getExtractor();
  const out = [];
  for (let i = 0; i < texts.length; i += 16) {
    const batch = await extractor(texts.slice(i, i + 16), { pooling: 'mean', normalize: true });
    out.push(...batch.tolist());
  }
  return out;
}
const cosineDistance = (a, b) => {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return 1 - dot; // vectors are normalized
};

// ---------------------------------------------------------------- ingest

const isMarkdown = (name) => /\.(md|markdown)$/i.test(name);
const isPdf = (name) => /\.pdf$/i.test(name);

async function ingestFiles(files, rebuild) {
  const t0 = performance.now();
  if (rebuild) index = [];
  let pdfCount = 0, mdCount = 0, skipped = 0;
  const newChunks = [];
  for (const { name, bytes, text } of files) {
    if (isPdf(name)) { pdfCount++; newChunks.push(...await parsePdf(name, bytes)); }
    else if (isMarkdown(name)) { mdCount++; newChunks.push(...parseMarkdown(name, text)); }
    else skipped++;
  }
  if (newChunks.length) {
    setStatus(`Embedding ${newChunks.length} segments…`);
    const vectors = await embed(newChunks.map((c) => c.text));
    newChunks.forEach((c, i) => { c.embedding = vectors[i]; });
    const replaced = new Set(newChunks.map((c) => c.file_name)); // re-ingesting a file replaces it
    index = index.filter((c) => !replaced.has(c.file_name)).concat(newChunks);
    await saveIndex();
  }
  return {
    total_files: pdfCount + mdCount,
    pdf_count: pdfCount,
    md_count: mdCount,
    skipped_count: skipped,
    chunk_count: newChunks.length,
    elapsed_seconds: (performance.now() - t0) / 1000,
    db_path: 'this browser (IndexedDB)',
  };
}

async function loadSampleNotes() {
  const manifest = await (await fetch(new URL('./sample_notes/manifest.json', import.meta.url))).json();
  const files = [];
  for (const name of manifest) {
    const res = await fetch(new URL(`./sample_notes/${encodeURIComponent(name)}`, import.meta.url));
    if (isPdf(name)) files.push({ name, bytes: new Uint8Array(await res.arrayBuffer()) });
    else files.push({ name, text: await res.text() });
  }
  return files;
}

// ---------------------------------------------------------------- retrieve (retrieve.py)

const tokenize = (text) => (text.toLowerCase().match(/\b\w+\b/g) || []);

class SimpleBM25 {
  constructor(corpus, k1 = 1.5, b = 0.75) {
    this.k1 = k1; this.b = b;
    this.corpusSize = corpus.length;
    this.termFreqs = []; this.docLengths = []; this.docFreqs = new Map();
    for (const doc of corpus) {
      const tokens = tokenize(doc);
      this.docLengths.push(tokens.length);
      const freqs = new Map();
      tokens.forEach((t) => freqs.set(t, (freqs.get(t) || 0) + 1));
      this.termFreqs.push(freqs);
      for (const t of freqs.keys()) this.docFreqs.set(t, (this.docFreqs.get(t) || 0) + 1);
    }
    this.avgDocLength = this.corpusSize ? this.docLengths.reduce((a, b) => a + b, 0) / this.corpusSize : 1;
  }
  score(query, i) {
    let score = 0;
    const freqs = this.termFreqs[i];
    for (const t of tokenize(query)) {
      if (!freqs.has(t)) continue;
      const tf = freqs.get(t);
      const df = this.docFreqs.get(t) || 0;
      const idf = Math.log(1 + (this.corpusSize - df + 0.5) / (df + 0.5));
      score += idf * (tf * (this.k1 + 1)) / (tf + this.k1 * (1 - this.b + this.b * (this.docLengths[i] / this.avgDocLength)));
    }
    return score;
  }
}

async function retrieveChunks(query, topK = 5, enableHybrid = true) {
  if (!index.length) return [];
  const [qv] = await embed([query]);
  const candidateK = Math.min(index.length, Math.max(topK * 3, 10));
  const candidates = index
    .map((c) => ({ ...c, distance: cosineDistance(qv, c.embedding) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, candidateK);

  if (enableHybrid && candidates.length > 1) {
    const bm25 = new SimpleBM25(candidates.map((c) => c.text));
    const scores = candidates.map((_, i) => bm25.score(query, i));
    const maxBm25 = Math.max(...scores) > 0 ? Math.max(...scores) : 1;
    candidates.forEach((c, i) => {
      const vecSim = Math.max(0, Math.min(1, 1 - c.distance));
      c.score = 0.7 * vecSim + 0.3 * (scores[i] / maxBm25);
    });
    candidates.sort((a, b) => b.score - a.score);
  } else {
    candidates.forEach((c) => { c.score = Math.max(0, 1 - c.distance); });
  }
  return candidates
    .filter((c) => c.distance <= DISTANCE_THRESHOLD || c.score > 0.4)
    .slice(0, topK);
}

// ---------------------------------------------------------------- generate (generate.py)

const uniqueSources = (chunks) => [...new Set(chunks.map(sourceCitation))];

function buildContextBlock(chunks) {
  return chunks.map((c, i) => {
    const lines = [`--- Excerpt ${i + 1} ---`, `File: ${c.file_name}`];
    if (c.doc_type === 'pdf' && c.page != null) lines.push(`Page: ${c.page}`);
    else if (c.doc_type === 'markdown' && c.heading) lines.push(`Section: "${c.heading}"`);
    lines.push(`Inline Citation: ${inlineCitationTag(c)}`, 'Content:', c.text.trim());
    return lines.join('\n');
  }).join('\n\n');
}
const userMessage = (query, chunks) =>
  `Context:\n${buildContextBlock(chunks)}\n\nQuestion: ${query}\n\nProvide your grounded answer with inline citations:`;

function generateOffline(query, chunks) {
  const stopwords = new Set(['what', 'did', 'i', 'write', 'about', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'for', 'to', 'in', 'of', 'and', 'how', 'do', 'does', 'can', 'tell', 'me', 'explain', 'why', 'when', 'which', 'where', 'my', 'notes', 'say', 'any', 'some', 'with', 'from', 'on']);
  const words = (s) => (s.match(/\b[a-zA-Z0-9_\-]+\b/g) || []).map((w) => w.toLowerCase());
  const queryWords = new Set(words(query).filter((w) => !stopwords.has(w) && w.length > 1));
  const scored = [];
  const seen = new Set();
  for (const chunk of chunks) {
    for (const s of chunk.text.split(/(?<=[.!?\n])\s+/)) {
      const clean = s.trim().replace(/^[-*• ]+/, '');
      if (clean.length < 15 || seen.has(clean)) continue;
      const overlap = new Set(words(clean).filter((w) => queryWords.has(w))).size;
      if (overlap > 0 || queryWords.size === 0) {
        scored.push({ score: overlap * 2 + (chunk.score || 0), sentence: clean, chunk });
        seen.add(clean);
      }
    }
  }
  if (!scored.length) return [NO_INFO, []];
  scored.sort((a, b) => b.score - a.score);
  const top = scored.slice(0, 4);
  const answer = ['Based on your notes:', ...top.map((t) => `- ${t.sentence} ${inlineCitationTag(t.chunk)}`)].join('\n');
  return [answer, uniqueSources(top.map((t) => t.chunk))];
}

const looksLikeNoInfo = (text) => /no relevant information found|could not find information|not enough information/i.test(text);

async function generateGemini(query, chunks, apiKey, model) {
  if (!apiKey) throw new Error('Add your Google Gemini API key in Settings, or switch to Zero API Key mode.');
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model || DEFAULT_GEMINI_MODEL)}:generateContent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
    body: JSON.stringify({
      system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
      contents: [{ role: 'user', parts: [{ text: userMessage(query, chunks) }] }],
      generationConfig: { temperature: 0.2, maxOutputTokens: 1024 },
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(`Gemini: ${data.error?.message || res.statusText}`);
  const answer = (data.candidates?.[0]?.content?.parts || []).map((p) => p.text || '').join('').trim();
  return [answer, looksLikeNoInfo(answer) ? [] : uniqueSources(chunks)];
}

async function generateAnthropic(query, chunks, apiKey, model) {
  if (!apiKey) throw new Error('Add your Anthropic API key in Settings, or switch to Zero API Key mode.');
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify({
      model: model || DEFAULT_ANTHROPIC_MODEL,
      max_tokens: 1024,
      system: SYSTEM_PROMPT,
      messages: [{ role: 'user', content: userMessage(query, chunks) }],
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(`Claude: ${data.error?.message || res.statusText}`);
  const answer = (data.content || []).map((b) => b.text || '').join('').trim();
  return [answer, looksLikeNoInfo(answer) ? [] : uniqueSources(chunks)];
}

// ---------------------------------------------------------------- /api/* routes

const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
const round3 = (x) => (x == null ? null : Math.round(x * 1000) / 1000);
const chunkView = (c) => ({
  chunk_id: c.chunk_id, text: c.text, file_name: c.file_name, doc_type: c.doc_type, page: c.page, heading: c.heading,
  source_citation: sourceCitation(c), inline_citation_tag: inlineCitationTag(c), score: round3(c.score), distance: round3(c.distance),
});

const routes = {
  'GET /api/status': async () => json({
    status: 'online', db_path: 'this browser (IndexedDB)', collection_name: COLLECTION_NAME, indexed_chunks: index.length,
    has_anthropic_key: false, has_gemini_key: false, default_provider: 'offline', default_model: DEFAULT_GEMINI_MODEL,
    gemini_model: DEFAULT_GEMINI_MODEL, anthropic_model: DEFAULT_ANTHROPIC_MODEL, public_mode: true,
  }),

  'POST /api/ingest': async (init) => {
    const { folder, rebuild } = JSON.parse(init.body || '{}');
    const name = String(folder || '').replace(/\\/g, '/').replace(/^\.?\//, '').replace(/\/$/, '');
    if (name !== 'sample_notes') {
      return json({ detail: "The browser version can't read folders on your computer. Use \"Upload files\" instead, or index ./sample_notes." }, 400);
    }
    return json({ success: true, stats: await ingestFiles(await loadSampleNotes(), !!rebuild) });
  },

  'POST /api/upload': async (init) => {
    const form = init.body instanceof FormData ? init.body : new FormData();
    const files = [];
    for (const f of form.getAll('files')) {
      if (isPdf(f.name)) files.push({ name: f.name, bytes: new Uint8Array(await f.arrayBuffer()) });
      else if (isMarkdown(f.name)) files.push({ name: f.name, text: await f.text() });
      else return json({ detail: `'${f.name}' is not a PDF or Markdown file.` }, 400);
    }
    if (!files.length) return json({ detail: 'No files uploaded.' }, 400);
    return json({ success: true, saved_files: files.map((f) => f.name), stats: await ingestFiles(files, false) });
  },

  'POST /api/retrieve': async (init) => {
    const req = JSON.parse(init.body || '{}');
    const chunks = await retrieveChunks(req.query, req.top_k || 5, req.enable_hybrid !== false);
    return json({ query: req.query, count: chunks.length, chunks: chunks.map(chunkView) });
  },

  'POST /api/ask': async (init) => {
    const req = JSON.parse(init.body || '{}');
    const chunks = await retrieveChunks(req.question, req.top_k || 5, req.enable_hybrid !== false);
    if (!chunks.length) return json({ answer: NO_INFO, sources: [], provider: req.provider || 'offline', chunks: [] });
    let prov = req.provider || 'offline';
    if (prov === 'auto') prov = req.api_key?.startsWith('AIza') ? 'gemini' : req.api_key?.startsWith('sk-ant') ? 'anthropic' : 'offline';
    try {
      const [answer, sources] = prov === 'gemini' ? await generateGemini(req.question, chunks, req.api_key, req.model)
        : prov === 'anthropic' ? await generateAnthropic(req.question, chunks, req.api_key, req.model)
          : generateOffline(req.question, chunks);
      return json({ answer, sources, provider: prov, chunks: chunks.map(chunkView) });
    } catch (e) {
      return json({ detail: e.message }, 400);
    }
  },
};

async function handle(url, init = {}) {
  const path = new URL(url, location.href).pathname.replace(/^.*(\/api\/)/, '/api/');
  const route = routes[`${(init.method || 'GET').toUpperCase()} ${path}`];
  if (!route) return json({ detail: `Unknown endpoint ${path}` }, 404);
  try {
    return await route(init);
  } catch (e) {
    console.error(e);
    return json({ detail: e.message || String(e) }, 500);
  }
}

// ---------------------------------------------------------------- UI touch-ups

function setStatus(text) {
  const el = document.getElementById('statusText');
  if (el) el.textContent = text;
}

function adaptUi() {
  const folder = document.getElementById('folderInput');
  if (folder) {
    folder.value = './sample_notes';
    folder.readOnly = true;
    folder.title = 'The browser version can index the bundled sample notes; use "Upload files" for your own.';
  }
  const offlineOption = document.querySelector('#modelSelect option[value^="ollama"]');
  if (offlineOption) offlineOption.remove();
  const banner = document.createElement('div');
  banner.className = 'text-center text-xs text-emerald-300 bg-emerald-950/40 border-b border-emerald-800/50 px-4 py-2';
  banner.innerHTML = '🔒 Runs entirely in your browser. Your notes, the search index and any API key stay on this device. '
    + '<a class="underline" href="https://github.com/Sege-Peter/Notes-QA" target="_blank" rel="noopener">Source on GitHub</a>';
  document.body.prepend(banner);
}

// ---------------------------------------------------------------- boot

window.__resolveNotesEngine({ handle });

(async () => {
  adaptUi();
  await loadIndex();
  if (typeof window.refreshStatus === 'function') window.refreshStatus();
  if (!index.length) {
    try {
      await ingestFiles(await loadSampleNotes(), false);
    } catch (e) {
      console.error(e);
      setStatus(`Could not prepare the demo: ${e.message}`);
      return;
    }
  }
  if (typeof window.refreshStatus === 'function') window.refreshStatus();
})();
