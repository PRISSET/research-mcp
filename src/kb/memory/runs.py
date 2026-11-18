"""Workflow lab: the runs the user's scripts recorded on disk, so a goal is answered from measured data.

A run script writes `results.json` into its output folder: {name: {exec_s, size, frames, seed, file, error?}}, with
`<name>.api.json` (the graph that ran) and the result file next to it. Each entry becomes a row of `runs` linked to
the catalog workflow whose file name it starts with, and a memory note (origin "run") so recall and workflows()
find it. The user rates runs (1-5 and a comment); ratings survive rescans.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from kb.memory import store, workflows
from kb.store import Database

RESULTS = "results.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def find(dirs: list[Path] | None = None) -> list[Path]:
    out = []
    for root in dirs or workflows.roots():
        for d, sub, files in os.walk(root):
            sub[:] = [x for x in sub if x not in workflows.SKIP_DIRS and not x.startswith(".")]
            if RESULTS in files:
                out.append(Path(d) / RESULTS)
    return out


def _entries(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items()
            if isinstance(v, dict) and ({"file", "exec_s", "error"} & set(v))}


def _workflow_for(db: Database, name: str) -> str | None:
    """The catalog workflow whose file stem the run name starts with (the longest such stem wins)."""
    best, best_len = None, 0
    for (ref,) in db.conn.execute("SELECT ref FROM memories WHERE origin = 'workflow'"):
        stem = Path(ref).stem.removesuffix(".api")
        if (name == stem or name.startswith(stem + "_")) and (best is None or len(stem) > best_len):
            best, best_len = ref, len(stem)
    return best


def _num(v: Any, kind: type) -> Any:
    try:
        return kind(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def describe(r: dict[str, Any]) -> str:
    parts = [f"Run {r['name']} in {r['folder']}" + (f" ({r['ran_at'][:10]})" if r.get("ran_at") else "") + "."]
    if r.get("workflow"):
        parts.append(f"Workflow: {r['workflow']}.")
    if r.get("error"):
        parts.append(f"Failed: {str(r['error'])[:200]}")
    else:
        facts = [f"{r['exec_s']:.0f} s" if r.get("exec_s") is not None else "", r.get("size") or "",
                 f"{r['frames']} frames" if r.get("frames") else "", f"seed {r['seed']}" if r.get("seed") is not None else ""]
        parts.append("Result: " + ", ".join(f for f in facts if f) + (f"; file {r['output']}" if r.get("output") else "") + ".")
    if r.get("rating"):
        parts.append(f"User rating {r['rating']}/5" + (f": {r['comment']}" if r.get("comment") else "") + ".")
    if r.get("settings"):
        parts.append(r["settings"].split("\n", 1)[1] if "\n" in r["settings"] else r["settings"])
    return "\n".join(parts)[:workflows.TEXT_CHARS]


def _note(db: Database, r: dict[str, Any]) -> None:
    text = describe(r)
    vec = store._vectors([f"Run {r['name']}\n{text}"]) if store._check_model(db) else None
    db.conn.execute("DELETE FROM memories WHERE origin = 'run' AND ref = ?", (r["key"],))
    db.conn.execute("INSERT INTO memories (topic, text, origin, ref, created, updated, emb) VALUES (?, ?, 'run', ?, ?, ?, ?)",
                    (f"Run: {Path(r['folder']).name}/{r['name']}", text, r["key"], _now(), _now(),
                     vec[0].tobytes() if vec is not None else None))


def scan(db: Database, dirs: list[Path] | None = None) -> dict[str, int]:
    """Record new runs and refresh changed ones; ratings stay."""
    changed = 0
    for path in find(dirs):
        folder = path.parent
        ran_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        for name, e in _entries(path).items():
            key = f"{path}#{name}"
            out_file = folder / str(e["file"]) if e.get("file") else None
            graph = folder / f"{name}.api.json"
            row = {"key": key, "name": name, "folder": str(folder),
                   "output": str(out_file) if out_file and out_file.exists() else None,
                   "graph": str(graph) if graph.exists() else None, "workflow": _workflow_for(db, name),
                   "exec_s": _num(e.get("exec_s"), float), "size": str(e["size"]) if e.get("size") else None,
                   "frames": _num(e.get("frames"), int), "seed": _num(e.get("seed"), int),
                   "error": str(e["error"])[:500] if e.get("error") else None,
                   "settings": workflows.describe(graph) if graph.exists() else None, "ran_at": ran_at}
            old = db.conn.execute("SELECT * FROM runs WHERE key = ?", (key,)).fetchone()
            if old and all(old[k] == row[k] for k in row):
                continue
            db.conn.execute(
                "INSERT INTO runs (key, name, folder, output, graph, workflow, exec_s, size, frames, seed, error, "
                "settings, ran_at, seen) VALUES (:key, :name, :folder, :output, :graph, :workflow, :exec_s, :size, "
                ":frames, :seed, :error, :settings, :ran_at, :seen) ON CONFLICT(key) DO UPDATE SET output = "
                "excluded.output, graph = excluded.graph, workflow = excluded.workflow, exec_s = excluded.exec_s, "
                "size = excluded.size, frames = excluded.frames, seed = excluded.seed, error = excluded.error, "
                "settings = excluded.settings, ran_at = excluded.ran_at, seen = excluded.seen",
                {**row, "seen": _now()})
            full = dict(db.conn.execute("SELECT * FROM runs WHERE key = ?", (key,)).fetchone())
            _note(db, full)
            changed += 1
    db.commit()
    return {"runs": db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0], "changed": changed}


def rate(db: Database, run: str, rating: int, comment: str | None = None) -> dict[str, Any]:
    """Rate a run 1-5 (by id, key or name; a name shared by several runs rates the latest)."""
    if not 1 <= int(rating) <= 5:
        raise ValueError("rating is 1..5")
    row = db.conn.execute("SELECT * FROM runs WHERE CAST(id AS TEXT) = ? OR key = ? OR name = ? "
                          "ORDER BY ran_at DESC LIMIT 1", (str(run), run, run)).fetchone()
    if row is None:
        raise ValueError(f"no run {run!r}: list them with workflows(goal) or `kb memory runs`")
    db.conn.execute("UPDATE runs SET rating = ?, comment = ? WHERE id = ?", (int(rating), comment, row["id"]))
    _note(db, dict(db.conn.execute("SELECT * FROM runs WHERE id = ?", (row["id"],)).fetchone()))
    db.commit()
    return {"run": row["name"], "id": row["id"], "rating": int(rating), "comment": comment}


def of_workflow(db: Database, path: str, limit: int = 5) -> list[dict[str, Any]]:
    """The runs of one catalog workflow: rated first, then the latest."""
    return [_brief(dict(r)) for r in db.conn.execute(
        "SELECT * FROM runs WHERE workflow = ? ORDER BY rating IS NULL, rating DESC, ran_at DESC LIMIT ?",
        (path, limit))]


def _brief(r: dict[str, Any]) -> dict[str, Any]:
    keep = ("id", "name", "exec_s", "size", "frames", "seed", "rating", "comment", "output", "error", "ran_at")
    return {k: r[k] for k in keep if r.get(k) is not None}


def listing(db: Database, limit: int = 50) -> list[dict[str, Any]]:
    return [_brief(dict(r)) | {"folder": r["folder"], "workflow": r["workflow"]} for r in db.conn.execute(
        "SELECT * FROM runs ORDER BY ran_at DESC, name LIMIT ?", (limit,))]
