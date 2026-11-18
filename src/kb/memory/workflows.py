"""A catalog of the ComfyUI workflows on this machine, so "which workflow did fast, good H3 clips with DPcache, and
where is it?" has an answer.

Each workflow JSON (UI or API format) under KB_WORKFLOW_DIRS becomes one memory note (origin "workflow", ref = its
path): the folder, the models it loads, the sampler settings, the speed-up nodes, its own Note nodes and the
folder README. A rescan reads only files whose mtime changed and drops notes of deleted files.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from kb.memory import store
from kb.store import Database

SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__", "site-packages", ".cache", "models", "output",
             "input", "tests", "fixtures", "blueprints", "temp"}
MAX_BYTES = 5_000_000
MODEL_FILE = re.compile(r"[\w.+-]+\.(?:safetensors|gguf|ckpt|pt|pth|bin|onnx|engine|sft)\b", re.IGNORECASE)
SPEED = re.compile(r"cache|sage|compile|teacache|swap|lightx|causvid|turbo|lcm|distill|fp8|fp4|nvfp|gguf|trt|"
                   r"tensorrt|sla|flash|step|fast|speed|quant|int8|tiled", re.IGNORECASE)
SAMPLER = re.compile(r"sampler|scheduler|sigmas", re.IGNORECASE)
NOTE = re.compile(r"note|markdown|comment|label", re.IGNORECASE)
SIZE = re.compile(r"width|height|length|frames|num_frames|steps|cfg|shift|denoise|fps", re.IGNORECASE)
NOTE_CHARS, TEXT_CHARS = 500, 1600


def roots() -> list[Path]:
    raw = os.environ.get("KB_WORKFLOW_DIRS") or os.pathsep.join(
        str(Path(p).expanduser()) for p in ("~/Documents", "~/Downloads", "~/Projects", "~/Desktop"))
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p]


def find(dirs: list[Path] | None = None) -> list[Path]:
    out = []
    for root in dirs or roots():
        for d, sub, files in os.walk(root):
            sub[:] = [x for x in sub if x not in SKIP_DIRS and not x.startswith(".")]
            run_dir = "results.json" in files   # its *.api.json are the graphs of recorded runs (see runs.py)
            out += [Path(d) / f for f in files if f.endswith(".json") and f != "results.json"
                    and not (run_dir and f.endswith(".api.json"))]
    return [p for p in out if _is_workflow(p)]


def _is_workflow(path: Path) -> bool:
    try:
        if not 50 <= path.stat().st_size <= MAX_BYTES:
            return False
        with open(path, "rb") as fh:
            head = fh.read(4000)
    except OSError:
        return False
    return b'"class_type"' in head or b'"nodes"' in head or b'"last_node_id"' in head   # describe() decides


def _nodes(data: Any) -> list[dict[str, Any]]:
    """Nodes as {type, title, values: {name or index: value}} from either format, subgraphs included."""
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        raw = list(data["nodes"])
        defs = data.get("definitions") if isinstance(data.get("definitions"), dict) else {}
        for sg in defs.get("subgraphs") or []:
            raw += sg.get("nodes") or [] if isinstance(sg, dict) else []
        out = []
        for n in raw:
            if not isinstance(n, dict):
                continue
            w = n.get("widgets_values")
            values = w if isinstance(w, dict) else dict(enumerate(w)) if isinstance(w, list) else {}
            out.append({"type": str(n.get("type", "")), "title": str(n.get("title") or ""), "values": values})
        return out
    if isinstance(data, dict):   # API format: {id: {class_type, inputs}}
        return [{"type": str(n.get("class_type", "")), "title": str((n.get("_meta") or {}).get("title") or ""),
                 "values": {k: v for k, v in (n.get("inputs") or {}).items() if not isinstance(v, list)}}
                for n in data.values() if isinstance(n, dict) and "class_type" in n]
    return []


def _short(v: Any) -> str:
    return str(v) if not isinstance(v, float) else f"{v:g}"


def describe(path: Path) -> str | None:
    """The note text of one workflow, or None when it is not readable JSON with nodes."""
    try:
        nodes = _nodes(json.loads(path.read_text(encoding="utf-8", errors="replace")))
    except (OSError, ValueError, TypeError, AttributeError):   # not a workflow after all, or a broken one
        return None
    if not nodes:
        return None
    types = list(dict.fromkeys(n["type"] for n in nodes if n["type"]))
    flat = " ".join(_short(v) for n in nodes for v in n["values"].values() if isinstance(v, str | int | float))
    models = list(dict.fromkeys(MODEL_FILE.findall(flat)))
    samplers = [f"{n['type']}[{', '.join(_short(v) for v in n['values'].values() if isinstance(v, str | int | float) and len(_short(v)) < 40)}]"
                for n in nodes if SAMPLER.search(n["type"])]
    sizes = list(dict.fromkeys(f"{k}={_short(v)}" for n in nodes for k, v in n["values"].items()
                               if isinstance(k, str) and SIZE.fullmatch(k) and isinstance(v, int | float)))
    notes = [str(v).strip() for n in nodes if NOTE.search(n["type"]) for v in n["values"].values()
             if isinstance(v, str) and len(v.strip()) > 20]
    speed = [t for t in types if SPEED.search(t)]
    readme = next((p for p in (path.parent / "README.md", path.parent / "readme.md") if p.exists()), None)
    parts = [f"Workflow {path.name} in {path.parent} (changed {datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d}).",
             f"Models: {', '.join(models[:12])}." if models else "",
             f"Sampling: {'; '.join(samplers[:4])}." if samplers else "",
             f"Settings: {', '.join(sizes[:14])}." if sizes else "",
             f"Speed-ups: {', '.join(speed[:12])}." if speed else "",
             f"Notes: {' | '.join(notes)[:NOTE_CHARS]}" if notes else "",
             f"Nodes: {', '.join(types[:30])}.",
             f"Folder README: {readme.read_text(encoding='utf-8', errors='replace')[:300].strip()}" if readme else ""]
    return "\n".join(p for p in parts if p)[:TEXT_CHARS]


def topic(path: Path) -> str:
    return f"Workflow: {path.parent.name}/{path.stem}"


def scan(db: Database, dirs: list[Path] | None = None) -> dict[str, int]:
    """Bring the catalog up to date: new and changed files are (re)described, deleted ones dropped."""
    found = find(dirs)
    known = {r[0][len("memory_wf:"):]: r[1] for r in db.conn.execute(
        "SELECT key, value FROM meta WHERE key LIKE 'memory_wf:%'")}
    ready = store._check_model(db)
    changed = 0
    for path in found:
        key, mtime = str(path), str(path.stat().st_mtime)
        if known.pop(key, None) == mtime:
            continue
        text = describe(path)
        db.conn.execute("DELETE FROM memories WHERE origin = 'workflow' AND ref = ?", (key,))
        if text:
            vec = store._vectors([f"{topic(path)}\n{text}"]) if ready else None
            now = datetime.now().isoformat(timespec="seconds")
            db.conn.execute("INSERT INTO memories (topic, text, origin, ref, created, updated, emb) "
                            "VALUES (?, ?, 'workflow', ?, ?, ?, ?)",
                            (topic(path), text, key, now, now, vec[0].tobytes() if vec is not None else None))
        db.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (f"memory_wf:{key}", mtime))
        changed += 1
    scanned = [r.resolve() for r in dirs or roots() if r.exists()]
    # gone: deleted, or under a scanned root but no longer a workflow; files under other roots keep their notes
    known = {k: v for k, v in known.items()
             if not Path(k).exists() or any(Path(k).resolve().is_relative_to(r) for r in scanned)}
    for gone in known:
        db.conn.execute("DELETE FROM memories WHERE origin = 'workflow' AND ref = ?", (gone,))
        db.conn.execute("DELETE FROM meta WHERE key = ?", (f"memory_wf:{gone}",))
    db.commit()
    n = db.conn.execute("SELECT COUNT(*) FROM memories WHERE origin = 'workflow'").fetchone()[0]
    return {"workflows": n, "changed": changed, "removed": len(known)}


def search(db: Database, goal: str, budget_tokens: int = 2000, rescan: bool = True) -> dict[str, Any]:
    """Workflows that fit the goal with their measured runs (time, size, rating, result file), the runs that fit
    it best, and what chats and RESEARCH.md recorded."""
    from kb.memory import runs

    scanned = {"catalog": scan(db), "runs": runs.scan(db)} if rescan else None
    store.sync_notebook(db)
    found = store.recall(db, goal, budget_tokens=budget_tokens // 2, per_topic=1, origins=("workflow",))
    for wf in found["notes"]:
        measured = runs.of_workflow(db, wf["path"])
        if measured:
            wf["runs"] = measured
    tried = store.recall(db, goal, budget_tokens=budget_tokens // 4, per_topic=1, origins=("run",))
    known = store.recall(db, goal, budget_tokens=budget_tokens // 4, origins=("chat", "notebook", "agent"))
    out: dict[str, Any] = {
        "goal": goal, "workflows": found["notes"], "runs": tried["notes"], "recorded": known["notes"],
        "next": "Name the best fitting workflow with its path and folder. Lead with measured runs (seconds, size, "
                "frames, the user's rating, result file) over descriptions; then propose concrete compromises — which "
                "speed-up (cache, fewer steps, distill LoRA, lower resolution then upscale) costs how much, with the "
                "numbers the runs give — and ask which way to go. When the user judges a result, save it with "
                "rate_run(run, rating 1-5, comment)."}
    if scanned:
        out["scanned"] = scanned
    return out
