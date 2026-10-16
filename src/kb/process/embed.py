"""Chunk embeddings with a static model (model2vec, no torch): ~130 MB of RAM, ~10k chunks/s on the CPU.

A static model is a lookup table of token vectors, so a page is embedded in milliseconds right when it is read —
no background queue. On the eval corpus (eval/) it lifts English recall@5 from 0.92 (BM25 alone) to 0.97, the
same as a 0.6B transformer at a fraction of the memory. It is English-only: the agent writes English queries.

Vectors are L2-normalised float16 blobs in chunks.emb; the model name lives in meta so a model change forces a
re-embed instead of silently mixing vector spaces.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from functools import lru_cache
from typing import Any

import numpy as np

from kb.store import Database

log = logging.getLogger(__name__)

MODEL = os.environ.get("KB_EMBED_MODEL", "minishlab/potion-retrieval-32M")
# Unload the model after this many idle seconds (0 = keep it): every agent session runs its own server.
IDLE_UNLOAD = float(os.environ.get("KB_MODEL_IDLE", "600"))

_lock = threading.Lock()
_model: Any = None
_last_use = 0.0
_timer: threading.Timer | None = None


def available() -> bool:
    from importlib.util import find_spec

    return find_spec("model2vec") is not None


def model() -> Any:
    """Loaded on first use, not at start (most sessions never search the base); local cache first."""
    global _model, _last_use
    with _lock:
        if _model is None:
            from model2vec import StaticModel

            try:
                from huggingface_hub import snapshot_download

                _model = StaticModel.from_pretrained(snapshot_download(MODEL, local_files_only=True))
            except Exception:   # first run: not downloaded yet
                _model = StaticModel.from_pretrained(MODEL)
        _last_use = time.monotonic()
        _schedule_unload()
        return _model


def _schedule_unload() -> None:
    global _timer
    if IDLE_UNLOAD <= 0 or (_timer is not None and _timer.is_alive()):
        return
    _timer = threading.Timer(IDLE_UNLOAD, _unload_if_idle)
    _timer.daemon = True
    _timer.start()


def _unload_if_idle() -> None:
    global _model, _timer
    with _lock:
        _timer = None
        if _model is None:
            return
        if time.monotonic() - _last_use < IDLE_UNLOAD:
            _schedule_unload()
            return
        _model = None
        _matrix.cache_clear()
    _trim()
    log.info("embedding model unloaded after %.0f s idle", IDLE_UNLOAD)


def _trim() -> None:
    """Hand freed memory back to the OS: glibc keeps it in the process otherwise."""
    import gc

    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):   # not glibc (macOS, musl, Windows)
        pass


def _encode(texts: list[str]) -> np.ndarray:
    v = np.asarray(model().encode(texts, show_progress_bar=False), dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def encode_docs(texts: list[str]) -> np.ndarray:
    return _encode(texts).astype(np.float16)


def encode_query(text: str) -> np.ndarray:
    return _encode([text])[0]


def _check_model(db: Database) -> bool:
    """True when the stored vectors came from another model (they are cleared and everything is re-embedded)."""
    row = db.conn.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
    changed = bool(row and row[0] != MODEL)
    if changed:
        log.warning("embedding model changed %s -> %s: re-embedding", row[0], MODEL)
        db.conn.execute("UPDATE chunks SET emb = NULL")
    db.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_model', ?)", (MODEL,))
    db.commit()
    return changed


def doc_text(section: str, text: str, title: str) -> str:
    return f"{title}\n{section}\n{text}"


def embed_pending(db: Database, limit: int | None = None, paper_ids: list[str] | None = None,
                  commit_every: int = 4096) -> dict[str, Any]:
    if _check_model(db):
        paper_ids = None
    sql = """SELECT c.id, c.section, c.text, p.title FROM chunks c JOIN papers p ON p.id = c.paper_id
             WHERE c.emb IS NULL"""
    params: list[str] = []
    if paper_ids is not None:
        sql += f" AND c.paper_id IN ({','.join('?' * len(paper_ids))})"
        params = list(paper_ids)
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = db.conn.execute(sql, params).fetchall()
    t0 = time.monotonic()
    for i in range(0, len(rows), commit_every):
        part = rows[i: i + commit_every]
        vecs = encode_docs([doc_text(r["section"], r["text"], r["title"]) for r in part])
        db.conn.executemany("UPDATE chunks SET emb = ? WHERE id = ?", [(v.tobytes(), r["id"]) for v, r in zip(vecs, part)])
        db.commit()
    _matrix.cache_clear()
    return {"embedded": len(rows), "model": MODEL, "seconds": round(time.monotonic() - t0, 2)}


def embed_now(db: Database, paper_ids: list[str]) -> str:
    """Vectors for freshly read documents. Never fails the read: keyword search works without them."""
    if not paper_ids:
        return "nothing to embed"
    if not available():
        return "unavailable (model2vec not installed): keyword search only"
    try:
        return f"{embed_pending(db, paper_ids=list(paper_ids))['embedded']} embedded"
    except Exception as e:
        log.warning("embedding failed: %s", e)
        return f"unavailable ({type(e).__name__}): keyword search only"


@lru_cache(maxsize=1)   # a new vector count (another session embedded something) replaces the old matrix
def _matrix(db_path: str, n_vectors: int) -> tuple[np.ndarray, np.ndarray]:
    """All vectors in one float16 matrix (cached per process; key includes the count so new vectors reload)."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, emb FROM chunks WHERE emb IS NOT NULL").fetchall()
    conn.close()
    return _stack(rows)


def _stack(rows: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 1), dtype=np.float16)
    return (np.array([r[0] for r in rows], dtype=np.int64),
            np.vstack([np.frombuffer(r[1], dtype=np.float16) for r in rows]))


def vector_search(db: Database, query: str, k: int = 100, paper_ids: set[str] | None = None) -> list[tuple[int, float]]:
    if not available():
        return []
    if paper_ids is not None:   # a few documents: score their vectors directly instead of loading the matrix
        ids, mat = _stack(db.conn.execute(
            f"SELECT id, emb FROM chunks WHERE emb IS NOT NULL AND paper_id IN ({','.join('?' * len(paper_ids))})",
            tuple(paper_ids)).fetchall())
    else:
        n = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE emb IS NOT NULL").fetchone()[0]
        ids, mat = _matrix(str(db.path), n) if n else _stack([])
    if len(ids) == 0:
        return []
    q = encode_query(query).astype(np.float16)
    if mat.shape[1] != q.shape[0]:   # vectors of another model: they are cleared on the next embed
        log.warning("stored vectors have %d dims, the model %d: run `kb base embed`", mat.shape[1], q.shape[0])
        return []
    scores = (mat @ q).astype(np.float32)
    top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    top = top[np.argsort(-scores[top])]
    return [(int(ids[i]), float(scores[i])) for i in top]
