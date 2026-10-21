"""Pack files: records, change files, snapshots and the manifest that lists them.

Every file is xz-compressed JSON lines. A record is `{"type": "doc" | "recipe", "id": …, …}`; its hash covers
everything but the hash itself, so an unchanged source gives the same hash and is not published again.
A change file holds `{"op": "put", "h": hash, …record}` and `{"op": "del", "type": …, "id": …}` lines; a snapshot
holds only puts. manifest.json per pack: the current version, the latest snapshot and the change files kept.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

FORMAT = 1
MANIFEST = "manifest.json"
DEFAULT_URL = "https://github.com/PRISSET/research-mcp/releases/download/packs"


def base_url() -> str:
    return (os.environ.get("KB_PACKS_URL") or DEFAULT_URL).rstrip("/")


def record_hash(rec: dict[str, Any]) -> str:
    body = {k: v for k, v in rec.items() if k not in ("h", "op")}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def file_name(pack: str, kind: str, version: int) -> str:
    return f"{pack}.{'snap' if kind == 'snapshot' else 'ch'}-{version:05d}.jsonl.xz"


def write_lines(path: Path, lines: Iterable[dict[str, Any]]) -> dict[str, Any]:
    raw = "".join(json.dumps(x, ensure_ascii=False, separators=(",", ":"), default=str) + "\n" for x in lines).encode()
    data = lzma.compress(raw, preset=9)
    path.write_bytes(data)
    return {"file": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "lines": raw.count(b"\n")}


def read_lines(data: bytes) -> Iterator[dict[str, Any]]:
    for line in lzma.decompress(data).splitlines():
        if line.strip():
            yield json.loads(line)


def get(base: str, name: str) -> bytes | None:
    """A published file: from the release (https) or a local folder (a local build, tests). None if missing."""
    if not base.startswith(("http://", "https://")):
        p = Path(base.removeprefix("file://")).expanduser() / name
        return p.read_bytes() if p.exists() else None
    import httpx

    from kb.config import settings

    r = httpx.get(f"{base}/{name}", follow_redirects=True, timeout=120, headers={"User-Agent": settings.user_agent})
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def load_manifest(base: str) -> dict[str, Any]:
    data = get(base, MANIFEST)
    return json.loads(data) if data else {"format": FORMAT, "packs": {}}


def verified(base: str, entry: dict[str, Any]) -> bytes:
    data = get(base, entry["file"])
    if data is None:
        raise RuntimeError(f"{entry['file']} is listed in the manifest but missing")
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"{entry['file']}: checksum mismatch (download corrupted or replaced)")
    return data


def files_to_apply(entry: dict[str, Any], local_version: int) -> tuple[str, list[dict[str, Any]]]:
    """("changes", the change files after local_version) when they are all kept and smaller than the snapshot;
    else ("snapshot", the snapshot + the change files after it)."""
    snap = entry["snapshot"]
    changes = sorted(entry.get("changes") or [], key=lambda c: c["version"])
    needed = [c for c in changes if c["version"] > local_version]
    if local_version and needed and [c["version"] for c in needed] == list(range(local_version + 1,
                                                                                 entry["version"] + 1)):
        if sum(c["bytes"] for c in needed) <= snap["bytes"]:
            return "changes", needed
    return "snapshot", [snap] + [c for c in changes if c["version"] > snap["version"]]


def replay(base: str, entry: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The published state of a pack, id -> record (the build compares the new records with it)."""
    if not entry:
        return {}
    state: dict[str, dict[str, Any]] = {}
    for f in files_to_apply(entry, 0)[1]:
        for line in read_lines(verified(base, f)):
            if line.get("op") == "del":
                state.pop(line["id"], None)
            else:
                state[line["id"]] = {k: v for k, v in line.items() if k != "op"}
    return state
