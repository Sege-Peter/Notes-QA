"""Build the static GitHub Pages version of the notes-qa web UI into docs/.

The page is the same HTML the FastAPI server serves; web/engine.js answers its
/api/* calls inside the browser. Run:  python web/build_site.py
"""
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from notes_qa.server import HTML_PAGE  # noqa: E402

OUT = ROOT / "docs"
PDFJS = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.7.76/build/"

# Installed before the page's own script runs: /api/* calls wait for the engine,
# everything else goes to the network as usual.
SHIM = """
  <script>
    (function () {
      var realFetch = window.fetch.bind(window);
      var ready = new Promise(function (resolve) { window.__resolveNotesEngine = resolve; });
      window.fetch = function (input, init) {
        var url = typeof input === 'string' ? input : (input instanceof URL ? input.href : input.url);
        if (url && url.indexOf('/api/') === 0) {
          return ready.then(function (engine) { return engine.handle(url, init || {}); });
        }
        return realFetch(input, init);
      };
    })();
  </script>
  <script type="module" src="engine.js"></script>
"""


def main() -> None:
    # empty docs/ in place (the folder itself may be held open by a local preview server)
    OUT.mkdir(exist_ok=True)
    for child in OUT.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    (OUT / "vendor").mkdir()

    html = HTML_PAGE.replace('<meta charset="UTF-8">', '<meta charset="UTF-8">' + SHIM, 1)
    assert "__resolveNotesEngine" in html, "could not inject the engine shim"
    html = html.replace("<title>", '<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 100 100%27%3E%3Ctext y=%27.9em%27 font-size=%2790%27%3E%F0%9F%93%9A%3C/text%3E%3C/svg%3E">\n  <title>', 1)
    html = html.replace("<title>", '<meta name="description" content="Ask questions over your PDFs and Markdown notes, with citations. Runs entirely in your browser.">\n  <title>', 1)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    shutil.copy(ROOT / "web" / "engine.js", OUT / "engine.js")

    samples = OUT / "sample_notes"
    shutil.copytree(ROOT / "sample_notes", samples)
    names = sorted(p.name for p in samples.iterdir() if p.suffix.lower() in {".pdf", ".md", ".markdown"})
    (samples / "manifest.json").write_text(json.dumps(names, indent=1), encoding="utf-8")

    # pdf.js is served from this site so its worker is same-origin
    for name in ("pdf.min.mjs", "pdf.worker.min.mjs"):
        with urllib.request.urlopen(PDFJS + name) as resp:
            (OUT / "vendor" / name).write_bytes(resp.read())

    (OUT / ".nojekyll").write_text("")
    print(f"built {OUT} ({len(names)} sample notes)")


if __name__ == "__main__":
    main()
