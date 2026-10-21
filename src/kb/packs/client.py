"""Install, update and remove knowledge packs (`uv run kb pack …`); the MCP server updates installed ones daily.

An update reads manifest.json (a few KB) and downloads only what the local version lacks: the change files after
it, or the snapshot when that is fewer bytes or the changes are no longer kept. Records whose hash matches the
local copy are skipped; the rest is written in one transaction, so an interrupted update leaves the previous
version. Passages get their numbers extracted and their vectors computed locally afterwards.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Any

from kb.packs import format as fmt
from kb.packs import recipes as rcp
from kb.store import Database

log = logging.getLogger(__name__)

CHECK_HOURS = 20
_lock = threading.Lock()


def installed(db: Database) -> dict[str, dict[str, Any]]:
    return {r["name"]: dict(r) for r in db.conn.execute("SELECT * FROM packs")}


def pack_ids(db: Database) -> set[str]:
    return {r[0] for r in db.conn.execute("SELECT item_id FROM pack_items WHERE type = 'doc'")}


def _put(db: Database, pack: str, rec: dict[str, Any]) -> None:
    rid = rec["id"]
    if rec["type"] == "recipe":
        rcp.store(db, rec)
    else:
        import json

        p = rec["paper"]
        db.conn.execute(
            """INSERT INTO papers (id, title, abstract, authors, year, published, url, source, kind, extra)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET title = excluded.title, abstract = excluded.abstract,
                   authors = excluded.authors, year = excluded.year, published = excluded.published,
                   url = excluded.url, source = excluded.source, kind = excluded.kind, extra = excluded.extra""",
            (rid, p["title"], p.get("abstract") or "", json.dumps(p.get("authors") or []), p.get("year"),
             p.get("published"), p.get("url"), p.get("source") or "pack", p.get("kind") or "web",
             json.dumps(p.get("extra") or {}, ensure_ascii=False)))
        db._reindex(rid)
        db.conn.execute("DELETE FROM chunks WHERE paper_id = ?", (rid,))
        db.conn.executemany(
            "INSERT INTO chunks (paper_id, ord, section, kind, page, text, n_words) VALUES (?, ?, ?, ?, NULL, ?, ?)",
            [(rid, i, s, k, t, len(t.split())) for i, (s, k, t) in enumerate(rec["chunks"])])
        db.conn.execute(
            """INSERT INTO paper_fulltext (paper_id, status, source, n_chars, n_chunks, used_at)
               VALUES (?, 'ok', ?, ?, ?, datetime('now'))
               ON CONFLICT(paper_id) DO UPDATE SET status = 'ok', source = excluded.source, n_chars = excluded.n_chars,
                   n_chunks = excluded.n_chunks, error = NULL, fetched_at = datetime('now')""",
            (rid, f"pack:{pack}", sum(len(c[2]) for c in rec["chunks"]), len(rec["chunks"])))
        if g := rec.get("grade"):
            db.conn.execute("INSERT OR REPLACE INTO critique (source_id, kind, grade, reasons, flags) "
                            "VALUES (?, ?, ?, ?, ?)", (rid, g["kind"], g["grade"], json.dumps(g["reasons"]),
                                                       json.dumps(g["flags"])))
    db.conn.execute("INSERT OR REPLACE INTO pack_items (pack, item_id, type, hash) VALUES (?, ?, ?, ?)",
                    (pack, rid, rec["type"], rec["h"]))


def _drop(db: Database, pack: str, rid: str, kind: str | None, cited: set[str]) -> None:
    db.conn.execute("DELETE FROM pack_items WHERE pack = ? AND item_id = ?", (pack, rid))
    if db.conn.execute("SELECT 1 FROM pack_items WHERE item_id = ?", (rid,)).fetchone():
        return   # another pack has it too
    if kind == "recipe" or rid.startswith("recipe:"):
        db.conn.execute("DELETE FROM recipes WHERE id = ?", (rid,))
    elif rid not in cited:   # the notebook's sources stay
        for table, col in (("critique", "source_id"), ("papers_fts", "paper_id"), ("papers", "id")):
            db.conn.execute(f"DELETE FROM {table} WHERE {col} = ?", (rid,))


def _cited() -> set[str]:
    try:
        from kb import notebook

        return notebook.cited_ids()
    except Exception:
        return set()


def _postprocess(db: Database, ids: list[str]) -> str:
    from kb.process.claims import extract_all
    from kb.process.embed import embed_now

    for i in range(0, len(ids), 400):
        extract_all(db, rebuild=True, paper_ids=ids[i:i + 400])
    db.commit()
    res = [embed_now(db, ids[i:i + 400]) for i in range(0, len(ids), 400)]
    return res[-1] if res else "nothing to embed"


def apply(db: Database, pack: str, files: list[bytes], mode: str, version: int) -> dict[str, Any]:
    own = {r[0]: r[1] for r in db.conn.execute("SELECT item_id, hash FROM pack_items WHERE pack = ?", (pack,))}
    kinds = {r[0]: r[1] for r in db.conn.execute("SELECT item_id, type FROM pack_items WHERE pack = ?", (pack,))}
    cited = _cited()
    seen: set[str] = set()
    docs: list[str] = []
    puts = dels = same = 0
    try:
        for data in files:
            for line in fmt.read_lines(data):
                rid = line["id"]
                if line.get("op") == "del":
                    if rid in own:
                        _drop(db, pack, rid, line.get("type"), cited)
                        own.pop(rid)
                        dels += 1
                    seen.discard(rid)
                    continue
                seen.add(rid)
                if own.get(rid) == line["h"]:
                    same += 1
                    continue
                _put(db, pack, line)
                own[rid] = line["h"]
                kinds[rid] = line["type"]
                puts += 1
                if line["type"] == "doc":
                    docs.append(rid)
        if mode == "snapshot":   # what the snapshot no longer has
            for rid in [r for r in own if r not in seen]:
                _drop(db, pack, rid, kinds.get(rid), cited)
                own.pop(rid)
                dels += 1
        db.conn.execute("INSERT OR REPLACE INTO packs (name, version, items, updated_at) "
                        "VALUES (?, ?, ?, datetime('now'))", (pack, version, len(own)))
        db.commit()
    except Exception:
        db.conn.rollback()
        raise
    return {"written": puts, "deleted": dels, "unchanged": same, "vectors": _postprocess(db, docs)}


def update(names: list[str] | None = None, base: str | None = None, install: bool = False) -> dict[str, Any]:
    """Bring installed packs (or `names`) to the published version; install=True adds packs not installed yet."""
    base = base or fmt.base_url()
    with _lock:
        manifest = fmt.load_manifest(base)
        db = Database()
        try:
            have = installed(db)
            targets = names or list(have)
            report: dict[str, Any] = {}
            for name in targets:
                entry = manifest["packs"].get(name)
                if entry is None:
                    report[name] = {"error": f"not published; available: {', '.join(manifest['packs']) or 'none'}"}
                    continue
                if name not in have and not install:
                    report[name] = {"error": "not installed: `uv run kb pack install " + name + "`"}
                    continue
                local = (have.get(name) or {}).get("version") or 0
                if local == entry["version"]:
                    report[name] = {"version": local, "status": "up to date"}
                    continue
                mode, files = fmt.files_to_apply(entry, local)
                data = [fmt.verified(base, f) for f in files]
                res = apply(db, name, data, mode, entry["version"])
                report[name] = {"from": local, "to": entry["version"], "downloaded": mode,
                                "files": [f["file"] for f in files], "bytes": sum(len(d) for d in data), **res}
            db.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('packs_checked', ?)",
                            (datetime.now().isoformat(timespec="minutes"),))
            db.commit()
            return report
        finally:
            db.close()


def remove(name: str) -> dict[str, Any]:
    db = Database()
    try:
        cited = _cited()
        rows = db.conn.execute("SELECT item_id, type FROM pack_items WHERE pack = ?", (name,)).fetchall()
        for rid, kind in rows:
            _drop(db, name, rid, kind, cited)
        db.conn.execute("DELETE FROM packs WHERE name = ?", (name,))
        db.commit()
        return {"pack": name, "removed": len(rows)}
    finally:
        db.close()


def status(base: str | None = None, online: bool = True) -> dict[str, Any]:
    db = Database()
    try:
        have = installed(db)
        counts = {r[0]: r[1] for r in db.conn.execute("SELECT pack, COUNT(*) FROM pack_items GROUP BY pack")}
    finally:
        db.close()
    out: dict[str, Any] = {"installed": {n: {"version": r["version"], "items": counts.get(n, 0),
                                             "updated": r["updated_at"]} for n, r in have.items()}}
    if online:
        try:
            man = fmt.load_manifest(base or fmt.base_url())
            out["available"] = {n: {k: e.get(k) for k in ("title", "description", "version", "docs", "recipes",
                                                          "updated")} | {"bytes": e["snapshot"]["bytes"]}
                                for n, e in man["packs"].items()}
        except Exception as e:
            out["available"] = {"error": f"manifest unavailable: {e}"}
    return out


def note() -> str:
    """For the server instructions: what is installed, so the agent searches it first."""
    try:
        db = Database()
        try:
            rows = db.conn.execute(
                "SELECT p.name, p.version, SUM(i.type = 'doc'), SUM(i.type = 'recipe') FROM packs p "
                "LEFT JOIN pack_items i ON i.pack = p.name GROUP BY p.name").fetchall()
        finally:
            db.close()
    except Exception:
        return ""
    if not rows:
        return ("\nKnowledge packs (trainer docs, issues and recipes of published weights for one field) are not "
                "installed. When the user trains LoRAs or fine-tunes LLMs, suggest once: `uv run kb pack install "
                "lora-training` / `llm-finetune` (or knowledge_packs(install=…) with their consent).\n")
    packs = ", ".join(f"{n} (v{v}: {d or 0} documents, {r or 0} recipes)" for n, v, d, r in rows)
    return (f"\nKnowledge packs installed: {packs}. For their field search them first — search_fulltext finds "
            "the trainers' docs and issue threads, recipes(model=…) gives the settings published weights used — "
            "then add live search for what is newer.\n")


def due(db: Database) -> bool:
    row = db.conn.execute("SELECT value FROM meta WHERE key = 'packs_checked'").fetchone()
    try:
        return datetime.now() - datetime.fromisoformat(row[0]) >= timedelta(hours=CHECK_HOURS)
    except (TypeError, ValueError):
        return True


def start_background() -> None:
    """At server start: update installed packs in a thread when the last check is older than CHECK_HOURS."""
    try:
        db = Database()
        try:
            if not installed(db) or not due(db):
                return
        finally:
            db.close()
    except Exception:
        return

    def go() -> None:
        try:
            log.info("packs: %s", update())
        except Exception as e:
            log.warning("pack update failed: %s", e)

    threading.Thread(target=go, name="packs", daemon=True).start()
