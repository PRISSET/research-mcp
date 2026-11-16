"""Notes by topic in the `memories` table: add (near-duplicates update instead), recall within a token budget,
mirror RESEARCH.md sections.

Search is hybrid like the paper base: FTS5 words (with prefixes, so Russian word forms meet) + vectors of a
multilingual static model (notes are mostly Russian, the paper base model is English-only), fused with RRF.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime
from typing import Any

import numpy as np

from kb.process import embed
from kb.store import Database

log = logging.getLogger(__name__)

EMBED_MODEL = os.environ.get("KB_MEMORY_EMBED_MODEL", "minishlab/potion-multilingual-128M")
SAME = 0.9            # cosine above which a new note in the same topic replaces the old one
RRF_K = 60
MIN_SIM = float(os.environ.get("KB_MEMORY_MIN_SIM", "0.25"))   # weaker vector matches are not hits at all
CHARS_PER_TOKEN = 3   # mixed Russian / English text
PIECE = 700           # notebook sections are cut into pieces of about this many characters
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _vectors(texts: list[str]) -> np.ndarray | None:
    """float16 vectors, or None when the model cannot be used (search then runs on words only)."""
    if not texts or not embed.available():
        return None
    try:
        return embed.encode_docs(texts, EMBED_MODEL)
    except Exception as e:   # no network on first use, broken cache
        log.warning("memory embedding failed: %s", e)
        return None


def _check_model(db: Database) -> bool:
    """True when every stored vector comes from EMBED_MODEL: after a model change all are re-made, and notes left
    without one (the model failed once) get theirs. While that cannot be done, vectors are not used at all."""
    row = db.conn.execute("SELECT value FROM meta WHERE key = 'memory_embed_model'").fetchone()
    same = bool(row and row[0] == EMBED_MODEL)
    sql = "SELECT id, topic, text FROM memories" + (" WHERE emb IS NULL" if same else "")
    rows = db.conn.execute(sql).fetchall()
    if rows:
        vecs = _vectors([f"{r['topic']}\n{r['text']}" for r in rows])
        if vecs is None:
            return same
        db.conn.executemany("UPDATE memories SET emb = ? WHERE id = ?",
                            [(v.tobytes(), r["id"]) for v, r in zip(vecs, rows)])
    if not same:
        db.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('memory_embed_model', ?)", (EMBED_MODEL,))
    db.commit()
    return True


def add(db: Database, topic: str, text: str, origin: str = "agent", ref: str | None = None,
        project: str | None = None) -> dict[str, Any]:
    topic, text = topic.strip(), text.strip()
    if not topic or not text:
        raise ValueError("topic and text are required")
    vec = _vectors([f"{topic}\n{text}"]) if _check_model(db) else None
    emb = vec[0].tobytes() if vec is not None else None
    old = _nearest(db, topic, text, vec)
    if old is not None:
        db.conn.execute("UPDATE memories SET text = ?, emb = ?, updated = ?, origin = ?, ref = ?, project = ? "
                        "WHERE id = ?", (text, emb, _now(), origin, ref, project, old))
        db.commit()
        return {"id": old, "action": "updated"}
    cur = db.conn.execute("INSERT INTO memories (topic, text, origin, ref, project, created, updated, emb) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (topic, text, origin, ref, project, _now(), _now(), emb))
    db.commit()
    return {"id": cur.lastrowid, "action": "added"}


def _nearest(db: Database, topic: str, text: str, vec: np.ndarray | None) -> int | None:
    """An existing note of the topic saying the same thing (same text, or a vector this close)."""
    rows = db.conn.execute("SELECT id, text, emb FROM memories WHERE topic = ? COLLATE NOCASE "
                           "AND origin NOT IN ('notebook', 'workflow')", (topic,)).fetchall()   # mirrors follow their file
    for r in rows:
        if r["text"].strip().lower() == text.lower():
            return r["id"]
    if vec is None:
        return None
    best, best_id = SAME, None
    for r in rows:
        if r["emb"]:
            other = np.frombuffer(r["emb"], dtype=np.float16)
            if other.shape == vec[0].shape:
                score = float(other.astype(np.float32) @ vec[0].astype(np.float32))
                if score >= best:
                    best, best_id = score, r["id"]
    return best_id


def forget(db: Database, ids: list[int]) -> int:
    n = db.conn.execute(f"DELETE FROM memories WHERE id IN ({','.join('?' * len(ids))})", ids).rowcount if ids else 0
    db.commit()
    return n


def topics(db: Database, limit: int = 40) -> list[dict[str, Any]]:
    return [dict(r) for r in db.conn.execute(
        "SELECT topic, COUNT(*) AS notes, MAX(updated) AS updated FROM memories GROUP BY topic COLLATE NOCASE "
        "ORDER BY MAX(updated) DESC LIMIT ?", (limit,))]


def _fts_query(text: str) -> str:
    """Each word also matches its other endings: `моделях` -> "модел"*."""
    terms = []
    for t in dict.fromkeys(re.findall(r"\w[\w.+-]*", text.lower())):
        if len(t) < 2:
            continue
        terms.append(f'"{t[:max(4, len(t) - 2)]}"*' if len(t) > 5 else f'"{t}"')
    return " OR ".join(terms) or '""'


def _where(topic: str | None, origins: tuple[str, ...] | None = None) -> tuple[str, list[Any]]:
    conds, params = [], []
    if topic:
        conds.append("m.topic = ? COLLATE NOCASE")
        params.append(topic)
    if origins:
        conds.append(f"m.origin IN ({','.join('?' * len(origins))})")
        params += list(origins)
    return ("WHERE " + " AND ".join(conds), params) if conds else ("", [])


def _words(db: Database, query: str, topic: str | None, k: int,
           origins: tuple[str, ...] | None = None) -> list[int]:
    where, params = _where(topic, origins)
    sql = (f"SELECT m.id FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid {where} "
           f"{'AND' if where else 'WHERE'} memories_fts MATCH ? ORDER BY bm25(memories_fts, 1.0, 0.5) LIMIT ?")
    try:
        return [r[0] for r in db.conn.execute(sql, [*params, _fts_query(query), k])]
    except Exception as e:   # malformed query
        log.debug("memory fts: %s", e)
        return []


def _meaning(db: Database, query: str, topic: str | None, k: int,
             origins: tuple[str, ...] | None = None) -> list[int]:
    q = _vectors([query]) if _check_model(db) else None
    if q is None:
        return []
    where, params = _where(topic, origins)
    rows = db.conn.execute(f"SELECT m.id, m.emb FROM memories m {where} {'AND' if where else 'WHERE'} "
                           "m.emb IS NOT NULL", params).fetchall()
    rows = [r for r in rows if len(r["emb"]) == q[0].nbytes]
    if not rows:
        return []
    mat = np.vstack([np.frombuffer(r["emb"], dtype=np.float16) for r in rows]).astype(np.float32)
    scores = mat @ q[0].astype(np.float32)
    return [rows[i]["id"] for i in np.argsort(-scores)[:k] if scores[i] >= MIN_SIM]


def recall(db: Database, query: str, topic: str | None = None, budget_tokens: int = 1200,
           per_topic: int = 5, origins: tuple[str, ...] | None = None) -> dict[str, Any]:
    """The notes that answer `query`, best first, until the budget is spent (a note too long for what is left is
    cut, so the first one is never dropped and the budget is never passed)."""
    fused: dict[int, float] = {}
    for ranking in (_words(db, query, topic, 50, origins), _meaning(db, query, topic, 50, origins)):
        for rank, mid in enumerate(ranking):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (RRF_K + rank + 1)
    order = sorted(fused, key=fused.get, reverse=True)
    rows = {r["id"]: r for r in db.conn.execute(
        f"SELECT id, topic, text, origin, ref, updated FROM memories WHERE id IN ({','.join('?' * len(order))})", order)
    } if order else {}
    out, used, per = [], 0, {}
    for mid in order:
        r = rows[mid]
        key = r["topic"].lower()
        extra = len(r["topic"]) + 40 + (len(r["ref"] or "") if r["origin"] == "workflow" else 0)
        room = budget_tokens * CHARS_PER_TOKEN - used - extra
        if per.get(key, 0) >= per_topic or room < 200:
            continue
        text = r["text"] if len(r["text"]) <= room else r["text"][:room - 2] + " …"
        per[key] = per.get(key, 0) + 1
        used += len(text) + extra
        out.append({"id": r["id"], "topic": r["topic"], "text": text, "origin": r["origin"],
                    "updated": r["updated"][:10]} | ({"path": r["ref"]} if r["origin"] == "workflow" else {}))
    result: dict[str, Any] = {"query": query, "notes": out, "tokens": used // CHARS_PER_TOKEN}
    if not out and not origins:
        result["topics"] = topics(db)
    return result


# ---------------- RESEARCH.md mirror ----------------

def _pieces(markdown: str) -> list[str]:
    """A section as pieces of about PIECE characters, cut at paragraph breaks."""
    text = COMMENT.sub("", markdown)
    out, cur = [], ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if cur and len(cur) + len(para) > PIECE:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
        while len(cur) > PIECE * 2:
            out.append(cur[:PIECE * 2])
            cur = cur[PIECE * 2:]
    if cur:
        out.append(cur)
    return out


def sync_notebook(db: Database) -> dict[str, int]:
    """Re-mirror the RESEARCH.md sections that changed since the last sync; drop removed ones."""
    from kb import notebook

    path = notebook.notebook_path()
    sections = notebook._sections(path.read_text(encoding="utf-8")) if path.exists() else []
    ready = _check_model(db)
    seen = {r[0][len("memory_nb:"):]: r[1] for r in db.conn.execute(
        "SELECT key, value FROM meta WHERE key LIKE 'memory_nb:%'")}
    changed = 0
    for s in sections:
        digest = hashlib.sha1(s["body"].encode()).hexdigest()
        if seen.pop(s["id"], None) == digest:
            continue
        pieces = _pieces(s["body"])
        vecs = _vectors([f"{s['title']}\n{p}" for p in pieces]) if ready else None
        db.conn.execute("DELETE FROM memories WHERE origin = 'notebook' AND ref = ?", (s["id"],))
        db.conn.executemany(
            "INSERT INTO memories (topic, text, origin, ref, created, updated, emb) VALUES (?, ?, 'notebook', ?, ?, ?, ?)",
            [(s["title"], p, s["id"], s.get("created") or _now(), s.get("updated") or _now(),
              vecs[i].tobytes() if vecs is not None else None) for i, p in enumerate(pieces)])
        db.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (f"memory_nb:{s['id']}", digest))
        changed += 1
    for gone in seen:
        db.conn.execute("DELETE FROM memories WHERE origin = 'notebook' AND ref = ?", (gone,))
        db.conn.execute("DELETE FROM meta WHERE key = ?", (f"memory_nb:{gone}",))
    db.commit()
    return {"sections": len(sections), "changed": changed, "removed": len(seen)}
