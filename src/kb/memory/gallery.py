"""A local gallery of workflow runs and the workflow catalog, served on 127.0.0.1 only (results can be NSFW: nothing
leaves the machine). Each run shows a poster frame and plays its video, time, size, frames, seed, the workflow it
came from and copy-path buttons; stars and a comment save a rating (the same as rate_run).

    uv run kb memory gallery          # opens http://127.0.0.1:3088
"""

from __future__ import annotations

import html
import json
import mimetypes
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from kb.memory import llm, runs, workflows
from kb.store import Database

PORT = 3088
CHUNK = 1 << 20


def _thumbs() -> Path:
    return llm.home() / "thumbs"


def thumbnail(run_id: int, output: str) -> Path | None:
    """A 360 px poster frame of the result (made once with ffmpeg, then cached per file version)."""
    st = Path(output).stat()
    out = _thumbs() / f"{run_id}-{st.st_mtime_ns}-{st.st_size}.jpg"
    if out.exists():
        return out
    src = Path(output)
    if src.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        return src
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "1", "-i", str(src), "-frames:v", "1",
                        "-vf", "scale=360:-2", str(out)], check=True, timeout=60, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out if out.exists() else None


def _data() -> dict[str, Any]:
    with Database() as db:
        rows = [dict(r) for r in db.conn.execute(
            "SELECT id, name, folder, output, workflow, exec_s, size, frames, seed, error, rating, comment, ran_at "
            "FROM runs ORDER BY ran_at DESC, name")]
        catalog = [{"path": r[0], "text": r[1]} for r in db.conn.execute(
            "SELECT ref, text FROM memories WHERE origin = 'workflow' ORDER BY ref")]
    return {"runs": rows, "workflows": catalog}


PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Workflow Lab</title><style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e2e2dc;--acc:#3a5bd9;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--mute:#9a9a92;--line:#2e2e2b;--acc:#8aa2ff;--bad:#ff7b6b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);padding:14px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;flex-wrap:wrap;align-items:center;z-index:2}
h1{font-size:18px;margin:0 12px 0 0}input[type=search]{flex:1;min-width:200px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}
.tabs button{padding:7px 12px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;cursor:pointer}.tabs button.on{border-color:var(--acc);color:var(--acc)}
main{padding:16px;max-width:1400px;margin:auto}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}
.media{aspect-ratio:16/9;background:#000;display:flex;align-items:center;justify-content:center;cursor:pointer;position:relative}
.media img,.media video{width:100%;height:100%;object-fit:contain}.media .play{position:absolute;font-size:34px;color:#fff;text-shadow:0 0 8px #000}
.body{padding:10px 12px;display:flex;flex-direction:column;gap:6px}.name{font-weight:600;word-break:break-all}
.facts{color:var(--mute);font-variant-numeric:tabular-nums}.err{color:var(--bad)}.stars span{cursor:pointer;font-size:20px;color:var(--line)}.stars span.on{color:#e5a50a}
textarea{width:100%;min-height:34px;resize:vertical;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);padding:6px;font:inherit}
.row{display:flex;gap:6px;flex-wrap:wrap}.row button{font-size:12px;padding:4px 8px;border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:6px;cursor:pointer}
.wf{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:8px}.wf pre{white-space:pre-wrap;color:var(--mute);margin:6px 0 0;font:12px/1.4 ui-monospace,monospace;max-height:7.5em;overflow:hidden}
.wf.open pre{max-height:none}.hidden{display:none!important}.note{color:var(--mute);margin:0 0 12px}
</style></head><body><header><h1>Workflow Lab</h1><input type="search" id="q" placeholder="Поиск: имя, папка, модель, узел, разрешение…">
<div class="tabs"><button data-t="runs" class="on">Прогоны</button><button data-t="wf">Workflow</button></div></header>
<main><section id="runs"><p class="note" id="rn"></p><div class="grid" id="rg"></div></section>
<section id="wf" class="hidden"><p class="note" id="wn"></p><div id="wl"></div></section></main>
<script>
const D=__DATA__;const $=s=>document.querySelector(s);const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const copy=t=>navigator.clipboard&&navigator.clipboard.writeText(t);
function card(r){const c=document.createElement("div");c.className="card";c.dataset.s=[r.name,r.folder,r.workflow,r.size,r.comment].join(" ").toLowerCase();
const facts=r.error?`<div class="err">${esc(r.error).slice(0,160)}</div>`:`<div class="facts">${r.exec_s?Math.round(r.exec_s)+" с":""} · ${esc(r.size||"")} · ${r.frames||"?"} кадров · seed ${r.seed??"?"}</div>`;
c.innerHTML=`<div class="media">${r.output?`<img loading="lazy" src="/thumb/${r.id}" alt=""><span class="play">▶</span>`:'<span class="facts">нет файла</span>'}</div>
<div class="body"><div class="name">${esc(r.name)}</div>${facts}<div class="facts">${esc((r.ran_at||"").slice(0,16).replace("T"," "))} · ${esc(r.folder.split("/").slice(-2).join("/"))}</div>
<div class="stars">${[1,2,3,4,5].map(i=>`<span data-v="${i}" class="${(r.rating||0)>=i?"on":""}">★</span>`).join("")}</div>
<textarea placeholder="Комментарий: что хорошо, что плохо">${esc(r.comment||"")}</textarea>
<div class="row">${r.output?'<button data-c="out">путь к видео</button>':""}${r.workflow?'<button data-c="wf">путь к workflow</button>':""}<button data-c="dir">папка</button></div></div>`;
const m=c.querySelector(".media");if(r.output)m.onclick=()=>{m.innerHTML=`<video controls autoplay src="/file/${r.id}"></video>`;m.onclick=null};
const ta=c.querySelector("textarea");const save=v=>fetch("/rate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:r.id,rating:v,comment:ta.value})}).then(x=>x.json()).then(j=>{if(j.error)alert(j.error);else{r.rating=v;c.querySelectorAll(".stars span").forEach(s=>s.classList.toggle("on",+s.dataset.v<=v))}});
c.querySelectorAll(".stars span").forEach(s=>s.onclick=()=>save(+s.dataset.v));ta.onchange=()=>{if(r.rating)save(r.rating)};
c.querySelectorAll("[data-c]").forEach(b=>b.onclick=()=>copy({out:r.output,wf:r.workflow,dir:r.folder}[b.dataset.c]));return c}
D.runs.forEach(r=>$("#rg").appendChild(card(r)));$("#rn").textContent=`${D.runs.length} прогонов. Клик по кадру — видео; звёзды сохраняют оценку (её видит агент).`;
D.workflows.forEach(w=>{const e=document.createElement("div");e.className="wf";e.dataset.s=(w.path+" "+w.text).toLowerCase();
e.innerHTML=`<div class="row"><span class="name">${esc(w.path.split("/").slice(-2).join("/"))}</span><button>путь</button></div><pre>${esc(w.text)}</pre>`;
e.querySelector("button").onclick=ev=>{ev.stopPropagation();copy(w.path)};e.onclick=()=>e.classList.toggle("open");$("#wl").appendChild(e)});
$("#wn").textContent=`${D.workflows.length} workflow на диске. Клик раскрывает описание.`;
$("#q").oninput=e=>{const q=e.target.value.toLowerCase().split(/\\s+/).filter(Boolean);document.querySelectorAll("[data-s]").forEach(x=>x.classList.toggle("hidden",!q.every(w=>x.dataset.s.includes(w))))};
document.querySelectorAll(".tabs button").forEach(b=>b.onclick=()=>{document.querySelectorAll(".tabs button").forEach(x=>x.classList.toggle("on",x===b));$("#runs").classList.toggle("hidden",b.dataset.t!=="runs");$("#wf").classList.toggle("hidden",b.dataset.t!=="wf")});
</script></body></html>"""


def page() -> str:
    data = json.dumps(_data(), ensure_ascii=False).replace("</", "<\\/")
    return PAGE.replace("__DATA__", data)


def _output_of(run_id: str) -> tuple[int, str] | None:
    """Only files recorded as run results are served: the id maps to a path from the database."""
    if not run_id.isdigit():
        return None
    with Database() as db:
        row = db.conn.execute("SELECT id, output FROM runs WHERE id = ?", (int(run_id),)).fetchone()
    return (row[0], row[1]) if row and row[1] and Path(row[1]).exists() else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._send(200, page().encode(), "text/html; charset=utf-8")
        kind, _, run_id = path.strip("/").partition("/")
        found = _output_of(run_id) if kind in ("thumb", "file") else None
        if found is None:
            return self._send(404, b"not found", "text/plain")
        if kind == "thumb":
            thumb = thumbnail(*found)
            if thumb is None:
                return self._send(404, b"no frame", "text/plain")
            return self._send(200, thumb.read_bytes(), mimetypes.guess_type(thumb.name)[0] or "image/jpeg")
        return self._file(Path(found[1]))

    def _file(self, path: Path) -> None:
        """The video, with Range support so the player can seek."""
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            a, _, b = rng[6:].split(",")[0].partition("-")
            if not a and b.isdigit():               # bytes=-N: the last N bytes
                start = max(0, size - int(b))
            else:
                start = int(a) if a.isdigit() else 0
                end = min(int(b), size - 1) if b.isdigit() else size - 1
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    block = fh.read(min(CHUNK, left))
                    if not block:
                        break
                    self.wfile.write(block)
                    left -= len(block)
            except (BrokenPipeError, ConnectionResetError):   # the player jumped elsewhere
                pass

    def do_POST(self) -> None:
        if self.path != "/rate":
            return self._send(404, b"not found", "text/plain")
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if self.headers.get("Content-Type") != "application/json" or (origin and origin != f"http://{host}"):
            return self._send(403, b"rate from the gallery page only", "text/plain")   # no cross-site writes
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with Database() as db:
                out = runs.rate(db, str(int(body["id"])), int(body["rating"]), (body.get("comment") or "").strip() or None)
        except (ValueError, KeyError, TypeError) as e:
            out = {"error": html.escape(str(e))}
        self._send(200, json.dumps(out, ensure_ascii=False).encode(), "application/json")


def serve(port: int = PORT, rescan: bool = True) -> ThreadingHTTPServer:
    if rescan:
        with Database() as db:
            workflows.scan(db)
            runs.scan(db)
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
