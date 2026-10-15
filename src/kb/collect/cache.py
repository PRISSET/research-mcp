"""On-disk HTTP cache: repeated research is instant and keyless rate limits (GitHub: 10 searches a minute) last.

Only successful responses are stored, compressed, in a separate SQLite file so the main base never waits on it.
Freshness depends on the source: searches change fast, papers and release files do not. KB_HTTP_CACHE=0 disables.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import time
import zlib
from urllib.parse import urlparse

from kb.config import settings

HOUR = 3600
# (host suffix, path prefix) -> seconds; first match wins.
TTL_RULES: list[tuple[str, str, int]] = [
    ("api.github.com", "/search", HOUR),
    ("api.github.com", "", 6 * HOUR),
    ("raw.githubusercontent.com", "", 6 * HOUR),
    ("huggingface.co", "/api/models?", 3 * HOUR),
    ("huggingface.co", "", 6 * HOUR),
    ("pypi.org", "", HOUR),
    ("api.openalex.org", "", 24 * HOUR),
    ("export.arxiv.org", "", 24 * HOUR),
    ("arxiv.org", "", 7 * 24 * HOUR),
    ("api.semanticscholar.org", "", 24 * HOUR),
    ("api2.openreview.net", "", 7 * 24 * HOUR),
    ("hn.algolia.com", "", 6 * HOUR),
    ("api.stackexchange.com", "", 6 * HOUR),
    ("html.duckduckgo.com", "", 6 * HOUR),
    ("api.search.brave.com", "", 6 * HOUR),
    ("api.tavily.com", "", 6 * HOUR),
]
DEFAULT_TTL = HOUR            # any other page read with fetch_url
MAX_BODY = 20 * 1024 * 1024


def enabled() -> bool:
    return os.environ.get("KB_HTTP_CACHE", "1") != "0"


def ttl_for(url: str) -> int:
    u = urlparse(url)
    target = u.path + ("?" + u.query if u.query else "")
    for host, prefix, ttl in TTL_RULES:
        if u.netloc.endswith(host) and target.startswith(prefix):
            return ttl
    return DEFAULT_TTL


def key_for(method: str, url: str, body: bytes | None) -> str:
    h = hashlib.sha256(f"{method.upper()} {url}".encode())
    if body:
        h.update(body)
    return h.hexdigest()


def _conn() -> sqlite3.Connection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.data_dir / "http_cache.sqlite", timeout=30)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS http_cache (
        key TEXT PRIMARY KEY, url TEXT NOT NULL, status INTEGER NOT NULL, headers TEXT NOT NULL,
        body BLOB NOT NULL, expires REAL NOT NULL)""")
    return conn


def get(key: str) -> tuple[int, dict[str, str], bytes] | None:
    try:
        conn = _conn()
        try:
            row = conn.execute("SELECT status, headers, body, expires FROM http_cache WHERE key = ?", (key,)).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if row is None or row[3] < time.time():
        return None
    return row[0], json.loads(row[1]), zlib.decompress(row[2])


def put(key: str, url: str, status: int, headers: dict[str, str], body: bytes, ttl: int) -> None:
    if len(body) > MAX_BODY or ttl <= 0:
        return
    keep = {k: v for k, v in headers.items() if k.lower() in ("content-type", "content-encoding-original", "link")}
    try:
        conn = _conn()
        try:
            conn.execute("INSERT OR REPLACE INTO http_cache VALUES (?, ?, ?, ?, ?, ?)",
                         (key, url[:2000], status, json.dumps(keep), zlib.compress(body, 6), time.time() + ttl))
            if random.random() < 0.02:   # occasional cleanup of expired entries
                conn.execute("DELETE FROM http_cache WHERE expires < ?", (time.time() - 7 * 24 * HOUR,))
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error:
        pass
