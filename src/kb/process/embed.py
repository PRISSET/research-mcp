"""Chunk embeddings (Qwen3-Embedding-0.6B, multilingual: Russian queries find English papers).

Vectors are L2-normalised float16 blobs in chunks.emb; the model name lives in meta so a model change
forces a re-embed instead of silently mixing vector spaces.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from functools import cache, lru_cache
from typing import Any

import numpy as np

from kb.store import Database

log = logging.getLogger(__name__)

MODEL = os.environ.get("KB_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
QUERY_PROMPT = "Instruct: Given a research question, retrieve passages from scientific papers that answer it\nQuery: "
MAX_SEQ = 640


MIN_FREE_GPU = 2 * 1024**3


def cpu_threads(background: bool = False) -> int:
    """The server runs next to the user's own work. Background embedding: a quarter of the cores (at most 4).
    A search the agent is waiting on: half of them (at most 8), for the few seconds it takes."""
    name = "KB_CPU_THREADS_BG" if background else "KB_CPU_THREADS"
    env = os.environ.get(name, "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    cores = os.cpu_count() or 4
    return max(1, min(4, cores // 4)) if background else max(1, min(8, cores // 2))


def use_cpu_threads() -> None:
    """torch keeps the intra-op thread count per calling thread: set it for the thread doing the work."""
    import torch

    torch.set_num_threads(cpu_threads(background=threading.current_thread().name == "embed"))


def pick_device() -> str:
    """CPU by default: the GPU belongs to the user's own work (generation, training), not to the research server.

    KB_DEVICE=cuda opts in; even then the GPU is used only with >= 2 GB free."""
    import torch

    want = os.environ.get("KB_DEVICE", "cpu").lower()
    if want != "cuda" or not torch.cuda.is_available():
        return "cpu"
    try:
        free, _ = torch.cuda.mem_get_info()
    except Exception:
        return "cpu"
    return "cuda" if free >= MIN_FREE_GPU else "cpu"


_LOAD_LOCK = threading.Lock()


def model(device: str | None = None) -> Any:
    # Parallel fetches embed at the same time: load the model once, not once per thread.
    with _LOAD_LOCK:
        touch()
        return _model(device or pick_device())


def load_local_first(cls: Any, name: str, **kw: Any) -> Any:
    """Cached weights load without asking the Hub (a dozen HEAD requests per model, and it works offline)."""
    try:
        return cls(name, local_files_only=True, **kw)
    except Exception:   # first run: not downloaded yet
        return cls(name, **kw)


@cache
def _model(device: str) -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    kw = {"torch_dtype": torch.float16} if device == "cuda" else {}
    m = load_local_first(SentenceTransformer, MODEL, device=device, model_kwargs=kw,
                         processor_kwargs={"padding_side": "left"})
    m.max_seq_length = MAX_SEQ
    return m


# ---------------- unloading when idle ----------------
# Every agent session starts its own server; one that sits idle should not keep ~1.5 GB of models in RAM.
IDLE_UNLOAD = float(os.environ.get("KB_MODEL_IDLE", "900"))
_last_use = time.monotonic()
_active = 0
_unloader: threading.Thread | None = None


class in_use:
    """Marks a model as busy for the length of an encode / rerank so it is not unloaded under it."""

    def __enter__(self) -> None:
        global _active
        with _cv:
            _active += 1
        touch()

    def __exit__(self, *exc: Any) -> None:
        global _active
        with _cv:
            _active -= 1
        touch()


def touch() -> None:
    global _last_use, _unloader
    _last_use = time.monotonic()
    if IDLE_UNLOAD > 0 and (_unloader is None or not _unloader.is_alive()):
        _unloader = threading.Thread(target=_unload_when_idle, name="unload", daemon=True)
        _unloader.start()


def _unload_when_idle() -> None:
    import gc

    while True:
        time.sleep(min(60.0, IDLE_UNLOAD / 2))
        if time.monotonic() - _last_use < IDLE_UNLOAD or _queue or _busy or _active:
            continue
        from kb.process import search

        with _LOAD_LOCK, search._LOAD_LOCK:
            loaded = _model.cache_info().currsize + search._reranker.cache_info().currsize
            _model.cache_clear()
            search._reranker.cache_clear()
            _matrix.cache_clear()
        if loaded:
            gc.collect()
            log.info("models unloaded after %.0f s idle", IDLE_UNLOAD)


def _encode(texts: list[str], batch_size: int) -> np.ndarray:
    import torch

    use_cpu_threads()
    with in_use():
        try:
            return model().encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False)
        except torch.OutOfMemoryError:  # another process took the GPU meanwhile
            log.warning("GPU out of memory: embedding on the CPU")
            torch.cuda.empty_cache()
            return model("cpu").encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                                      show_progress_bar=False)


def encode_docs(texts: list[str], batch_size: int = 32) -> np.ndarray:
    return _encode(texts, batch_size).astype(np.float16)


def encode_query(text: str) -> np.ndarray:
    return _encode([QUERY_PROMPT + text], 1)[0].astype(np.float32)


def _check_model(db: Database) -> None:
    row = db.conn.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
    if row and row[0] != MODEL:
        log.warning("embedding model changed %s -> %s: clearing vectors", row[0], MODEL)
        db.conn.execute("UPDATE chunks SET emb = NULL")
    db.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_model', ?)", (MODEL,))
    db.commit()


def doc_text(section: str, text: str, title: str) -> str:
    return f"{title}\n{section}\n{text}"


def embed_pending(db: Database, batch_size: int = 32, limit: int | None = None, commit_every: int = 2048,
                  paper_ids: list[str] | None = None) -> dict[str, Any]:
    _check_model(db)
    sql = """SELECT c.id, c.section, c.text, p.title FROM chunks c JOIN papers p ON p.id = c.paper_id
             WHERE c.emb IS NULL"""
    params: list[str] = []
    if paper_ids is not None:
        sql += f" AND c.paper_id IN ({','.join('?' * len(paper_ids))})"
        params = list(paper_ids)
    sql += " ORDER BY length(c.text)"  # length-sorted batches waste less padding
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = db.conn.execute(sql, params).fetchall()
    t0, done = time.monotonic(), 0
    for i in range(0, len(rows), commit_every):
        part = rows[i: i + commit_every]
        vecs = encode_docs([doc_text(r["section"], r["text"], r["title"]) for r in part], batch_size)
        db.conn.executemany("UPDATE chunks SET emb = ? WHERE id = ?", [(v.tobytes(), r["id"]) for v, r in zip(vecs, part)])
        db.commit()
        done += len(part)
        log.info("embedded %d/%d (%.0f chunks/s)", done, len(rows), done / (time.monotonic() - t0))
    _matrix.cache_clear()
    return {"embedded": done, "model": MODEL, "seconds": round(time.monotonic() - t0, 1)}


# ---------------- background embedding ----------------
# Reading a page must not block the agent (or burn every core) while its chunks are embedded: fetch_url and
# fetch_fulltext queue the ids and return; one worker thread embeds them. Until then BM25 still finds the text.
_queue: set[str] = set()
_busy = False
_cv = threading.Condition()
_worker: threading.Thread | None = None


def _persist(db: Database, add: list[str] = (), remove: list[str] = ()) -> None:  # type: ignore[assignment]
    """The queue survives a restart: the agent may close right after reading a page."""
    db.conn.execute("CREATE TABLE IF NOT EXISTS embed_queue (paper_id TEXT PRIMARY KEY)")
    db.conn.executemany("INSERT OR IGNORE INTO embed_queue VALUES (?)", [(i,) for i in add])
    db.conn.executemany("DELETE FROM embed_queue WHERE paper_id = ?", [(i,) for i in remove])
    db.commit()


def resume_queue() -> str:
    db = Database()
    try:
        _persist(db)
        left = [r[0] for r in db.conn.execute("SELECT paper_id FROM embed_queue")]
    finally:
        db.close()
    return embed_later(left, persist=False) if left else "nothing to embed"


def embed_later(paper_ids: list[str], persist: bool = True) -> str:
    global _worker
    if not paper_ids:
        return "nothing to embed"
    db = Database()
    try:
        if os.environ.get("KB_EMBED_SYNC") == "1":
            return f"{embed_pending(db, paper_ids=list(paper_ids))['embedded']} embedded"
        if persist:
            _persist(db, add=list(paper_ids))
    finally:
        db.close()
    with _cv:
        _queue.update(paper_ids)
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_work, name="embed", daemon=True)
            _worker.start()
        _cv.notify()
    return "queued (embedding in the background; keyword search works meanwhile)"


def _work() -> None:
    global _busy
    while True:
        with _cv:
            while not _queue:
                _cv.wait()
            ids = sorted(_queue)
            _queue.clear()
            _busy = True
        try:
            db = Database()
            try:
                embed_pending(db, paper_ids=ids)
                _persist(db, remove=ids)
            finally:
                db.close()
        except Exception as e:  # no torch, or the base is busy: keyword search still works
            log.warning("background embedding failed: %s", e)
        finally:
            with _cv:
                _busy = False
                _cv.notify_all()


def wait_idle(timeout: float) -> bool:
    """Let a search see vectors for pages read a moment ago, but never wait longer than `timeout`."""
    deadline = time.monotonic() + timeout
    with _cv:
        while _queue or _busy:
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            _cv.wait(left)
    return True


@lru_cache(maxsize=1)   # a new vector count (another session embedded something) replaces the old matrix
def _matrix(db_path: str, n_vectors: int) -> tuple[np.ndarray, np.ndarray]:
    """All vectors in one float16 matrix (cached per process; key includes the count so new vectors reload)."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    ids, vecs = [], []
    for cid, blob in conn.execute("SELECT id, emb FROM chunks WHERE emb IS NOT NULL"):
        ids.append(cid)
        vecs.append(np.frombuffer(blob, dtype=np.float16))
    conn.close()
    if not vecs:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 1), dtype=np.float16)
    return np.array(ids, dtype=np.int64), np.vstack(vecs)


def vector_search(db: Database, query: str, k: int = 100, paper_ids: set[str] | None = None) -> list[tuple[int, float]]:
    n = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE emb IS NOT NULL").fetchone()[0]
    if n == 0:
        return []
    ids, mat = _matrix(str(db.path), n)
    q = encode_query(query).astype(np.float16)
    scores = (mat @ q).astype(np.float32)
    if paper_ids is not None:
        allowed = {r[0] for r in db.conn.execute(
            f"SELECT id FROM chunks WHERE paper_id IN ({','.join('?' * len(paper_ids))})", tuple(paper_ids))}
        mask = np.isin(ids, np.fromiter(allowed, dtype=np.int64))
        scores = np.where(mask, scores, -1.0)
    top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    top = top[np.argsort(-scores[top])]
    return [(int(ids[i]), float(scores[i])) for i in top if scores[i] > -1.0]
