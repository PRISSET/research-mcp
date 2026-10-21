"""Build a pack version: collect the records, compare with the published ones, write only the difference.

Sources of a pack are listed in defs/<pack>.yaml: trainer docs and example configs from GitHub (one tree request
per repo; a file whose blob sha did not change is not downloaded again), the repos' README with release notes,
their most reacted issues with all comments (re-read only when updated), model cards that describe training, and
recipes from Hugging Face and Civitai weights (a header is read once per repo revision / Civitai version: the
build cache remembers it). Documents are read with the same readers as fetch(), into a scratch base, and
exported with their passages.

If a source fails (rate limit, outage) its previous records are kept, and a build that would delete more than a
third of the pack stops: a broken source must not empty the users' packs.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from kb.config import settings
from kb.packs import format as fmt
from kb.packs import recipes as rcp
from kb.store import Database

log = logging.getLogger(__name__)

DEFS = Path(__file__).with_name("defs")
SNAPSHOT_EVERY = 30      # versions between snapshots
KEEP_CHANGES = 60        # change files kept: a client up to two months behind still downloads only the changes
MAX_DELETE_SHARE = 0.34
VOLATILE = {"pushed", "age_months", "used_at"}
ROUNDED = {"stars", "forks", "downloads", "likes", "open_issues", "comments_n", "dislikes"}


def available() -> list[str]:
    return sorted(p.stem for p in DEFS.glob("*.yaml"))


def load_def(name: str) -> dict[str, Any]:
    path = DEFS / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown pack {name!r}; known: {', '.join(available())}")
    return {"name": name, **yaml.safe_load(path.read_text(encoding="utf-8"))}


# ---------------- documents ----------------

def export_doc(db: Database, doc_id: str, src: str, rev: str | None) -> dict[str, Any] | None:
    """A document of the scratch base as a pack record: metadata, grade and passages."""
    p = db.conn.execute("SELECT title, abstract, authors, year, published, url, source, kind, extra FROM papers "
                        "WHERE id = ?", (doc_id,)).fetchone()
    if not p:
        return None
    chunks = [[c["section"], c["kind"], c["text"]] for c in db.conn.execute(
        "SELECT section, kind, text FROM chunks WHERE paper_id = ? ORDER BY ord", (doc_id,))]
    if not chunks:
        return None
    extra = {k: (rcp._round2(v) if k in ROUNDED else v) for k, v in json.loads(p["extra"] or "{}").items()
             if k not in VOLATILE}
    g = db.conn.execute("SELECT kind, grade, reasons, flags FROM critique WHERE source_id = ?", (doc_id,)).fetchone()
    rec = {"type": "doc", "id": doc_id, "src": src, "rev": rev,
           "paper": {"title": p["title"], "abstract": p["abstract"], "authors": json.loads(p["authors"] or "[]"),
                     "year": p["year"], "published": p["published"], "url": p["url"], "source": p["source"],
                     "kind": p["kind"], "extra": extra},
           "grade": {"kind": g["kind"], "grade": g["grade"], "reasons": json.loads(g["reasons"]),
                     "flags": json.loads(g["flags"])} if g else None,
           "chunks": chunks}
    return {k: v for k, v in rec.items() if v is not None}


def _tree(http: Any, repo: str) -> list[dict[str, Any]]:
    from kb.research.platforms import GH_API

    data = http.get(f"{GH_API}/repos/{repo}/git/trees/HEAD", params={"recursive": "1"}).json()
    return [t for t in data.get("tree") or [] if t.get("type") == "blob"]


def collect_docs(spec: dict[str, Any], prev: dict[str, dict[str, Any]], db: Database,
                 out: dict[str, dict[str, Any]], failed: set[str]) -> None:
    from kb.research.platforms import _gh, github_read

    exclude = re.compile(spec.get("exclude") or r"$^")
    cap = int(spec.get("max_files") or 150)
    http = _gh()
    try:
        for d in spec.get("docs") or []:
            repo = d["repo"]
            src = f"docs:{repo}"
            try:
                files = [t for t in _tree(http, repo) if any(fnmatch.fnmatch(t["path"], g) for g in d["include"])
                         and not exclude.search(t["path"]) and (t.get("size") or 0) < 400_000]
                files.sort(key=lambda t: (t["path"].count("/"), t["path"]))
                for t in files[:cap]:
                    did = f"gh:{repo.lower()}/{t['path']}"
                    if prev.get(did, {}).get("rev") == t["sha"]:
                        out[did] = prev[did]
                        continue
                    github_read(db, repo, path=t["path"])
                    if rec := export_doc(db, did, src, t["sha"]):
                        out[did] = rec
                github_read(db, repo)   # README + recent release notes
                if rec := export_doc(db, f"gh:{repo.lower()}", src, None):
                    out[rec["id"]] = rec
            except Exception as e:
                log.warning("pack docs %s failed: %s", repo, e)
                failed.add(src)
    finally:
        http.close()


def collect_issues(spec: dict[str, Any], prev: dict[str, dict[str, Any]], db: Database,
                   out: dict[str, dict[str, Any]], failed: set[str]) -> None:
    from kb.research.platforms import GH_API, _gh, github_read

    conf = spec.get("issues") or {}
    http = _gh()
    try:
        for repo in conf.get("repos") or []:
            src = f"issues:{repo}"
            try:
                found = []
                for sort in ("reactions", "comments"):
                    data = http.get(f"{GH_API}/search/issues", params={
                        "q": f"repo:{repo} is:issue comments:>2", "sort": sort, "order": "desc",
                        "per_page": int(conf.get("per_repo") or 15)}).json()
                    found += data.get("items") or []
                for i in {i["number"]: i for i in found}.values():
                    did = f"gh:{repo.lower()}#{i['number']}"
                    if prev.get(did, {}).get("rev") == i.get("updated_at"):
                        out[did] = prev[did]
                        continue
                    github_read(db, repo, issue=i["number"])
                    if rec := export_doc(db, did, src, i.get("updated_at")):
                        out[did] = rec
            except Exception as e:
                log.warning("pack issues %s failed: %s", repo, e)
                failed.add(src)
    finally:
        http.close()


CARD_WORDS = re.compile(r"(?i)learning.?rate|\blr\b|\bepochs?\b|\bsteps\b|dataset|trained (on|with)|training "
                        r"(details|settings|data|config|parameters)|network.?dim|\brank\b|optimizer|captions?")


def collect_cards(spec: dict[str, Any], models: list[dict[str, Any]], prev: dict[str, dict[str, Any]],
                  db: Database, out: dict[str, dict[str, Any]], failed: set[str]) -> None:
    """Cards of the most downloaded weights with recipes, when the card talks about how they were trained."""
    from kb.research.platforms import hf_read

    n = int(spec.get("cards") or 0)
    done = 0
    for m in models:
        if done >= n:
            break
        did = f"hf:{m['id'].lower()}"
        if prev.get(did, {}).get("rev") == m.get("sha"):
            out[did] = prev[did]
            done += 1
            continue
        try:
            card = hf_read(db, m["id"]).get("card") or ""
        except Exception as e:
            log.info("card %s: %s", m["id"], e)
            continue
        if len(set(x.group(0).lower() for x in CARD_WORDS.finditer(card))) >= 3:
            if rec := export_doc(db, did, "cards", m.get("sha")):
                out[did] = rec
                done += 1


# ---------------- recipes ----------------

def _read_new(models: list[Any], recipe: Any, rid: Any, known: Any, budget: int, workers: int,
              prev: dict[str, dict[str, Any]], out: dict[str, dict[str, Any]], src: str) -> list[Any]:
    """Recipes for `models` (most popular first): already read ones from the cache, at most `budget` new downloads
    (the rest waits for the next build), a temporary failure keeps the published record. Returns the models that
    gave a recipe."""
    got: list[Any] = []
    retries = 0

    def keep(m: Any) -> None:
        if rid(m) in prev:
            out[rid(m)] = prev[rid(m)]

    new = []
    for m in models:
        if known(m):
            if rec := recipe(m):
                out[rec["id"]] = {**rec, "src": src}
                got.append(m)
        elif len(new) < budget:
            new.append(m)
        else:
            keep(m)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(new), workers):
            part = new[i:i + workers]
            if retries >= 12:   # rate limited: stop asking, the next build continues
                for m in part:
                    keep(m)
                continue
            for m, fut in [(m, pool.submit(recipe, m)) for m in part]:
                try:
                    rec = fut.result()
                except rcp.Retry:
                    retries += 1
                    keep(m)
                    continue
                if rec:
                    out[rec["id"]] = {**rec, "src": src}
                    got.append(m)
    if retries:
        log.warning("%s: %d files hit rate limits, kept for the next build", src, retries)
    return got


def _hf_listed(http: Any, hf: dict[str, Any], llm: bool) -> list[dict[str, Any]]:
    """Hugging Face repos of every filter x sort, most downloaded first; for images only diffusion ones."""
    seen: dict[str, dict[str, Any]] = {}
    for filt in hf.get("filters") or ["lora"]:
        for sort in hf.get("sorts") or ["downloads"]:
            for m in rcp.hf_models(http, filt, sort, int(hf.get("limit") or 200)):
                image = m.get("pipeline_tag") in rcp.IMAGE_PIPES or \
                    bool({"diffusers", "stable-diffusion", "text-to-image"} & set(m.get("tags") or []))
                if llm or image:
                    seen.setdefault(m["id"], m)
    return sorted(seen.values(), key=lambda m: m.get("downloads") or 0, reverse=True)


def _civitai_listed(api: Any, cv: dict[str, Any]) -> list[dict[str, Any]]:
    items: dict[int, dict[str, Any]] = {}
    for period in cv.get("periods") or ["AllTime"]:
        for m in rcp.civitai_models(api, period, int(cv.get("pages") or 2)):
            items.setdefault(m["id"], m)
    return sorted(items.values(), key=lambda m: (m.get("stats") or {}).get("downloadCount") or 0, reverse=True)


def collect_recipes(spec: dict[str, Any], cache: dict[str, Any], prev: dict[str, dict[str, Any]],
                    out: dict[str, dict[str, Any]], failed: set[str]) -> list[dict[str, Any]]:
    """Recipe records; returns the Hugging Face repos that had one, most downloaded first (for their cards)."""
    from kb.collect.http import RateLimitedClient

    conf = spec.get("recipes") or {}
    llm = bool(conf.get("llm"))
    with_recipe: list[dict[str, Any]] = []
    hf = conf.get("hf") or {}
    if hf:
        http = RateLimitedClient(0.05, max_retries=2, max_backoff=20, use_cache=False, timeout=60)
        try:
            models = _hf_listed(http, hf, llm)
            with_recipe = _read_new(models, lambda m: rcp.hf_recipe(http, m, llm, cache),
                                    lambda m: f"recipe:hf:{m['id'].lower()}", lambda m: rcp.known(m, cache),
                                    int(hf.get("new_per_build") or 800), 8, prev, out, "recipes:hf")
        except Exception as e:
            log.warning("pack recipes hf failed: %s", e)
            failed.add("recipes:hf")
        finally:
            http.close()
    cv = conf.get("civitai") or {}
    if cv:
        # Civitai limits downloads hard (429 after a few parallel requests): one at a time, a second apart
        http = RateLimitedClient(1.0, max_retries=1, max_backoff=10, use_cache=False, timeout=60)
        try:
            from kb.research.platforms import _civitai_client

            api = _civitai_client()
            try:
                models = _civitai_listed(api, cv)
            finally:
                api.close()
            _read_new(models, lambda m: rcp.civitai_recipe(http, m, cache),
                      lambda m: f"recipe:civitai:{(m.get('modelVersions') or [{}])[0].get('id')}",
                      lambda m: rcp.known(m, cache, civitai=True), int(cv.get("new_per_build") or 150), 1, prev,
                      out, "recipes:civitai")
        except Exception as e:
            log.warning("pack recipes civitai failed: %s", e)
            failed.add("recipes:civitai")
        finally:
            http.close()
    return sorted(with_recipe, key=lambda m: m.get("downloads") or 0, reverse=True)


# ---------------- versions ----------------

def _cache_path(name: str) -> Path:
    return settings.data_dir / "pack-build" / f"{name}.cache.json"


def collect(spec: dict[str, Any], prev: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    out: dict[str, dict[str, Any]] = {}
    failed: set[str] = set()
    cpath = _cache_path(spec["name"])
    try:
        cache = json.loads(cpath.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "build.sqlite")
        try:
            try:
                models = collect_recipes(spec, cache, prev, out, failed)
            finally:   # what was read is remembered even if the build stops later
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cpath.write_text(json.dumps(cache), encoding="utf-8")
            collect_docs(spec, prev, db, out, failed)
            collect_issues(spec, prev, db, out, failed)
            collect_cards(spec, models, prev, db, out, failed)
        finally:
            db.close()
    for rid, rec in prev.items():   # a source that failed keeps what it had
        if rid not in out and (rec.get("src") in failed or (rec.get("src") == "cards" and "recipes:hf" in failed)):
            out[rid] = rec
    return out, failed


def publish(name: str, spec: dict[str, Any], records: dict[str, dict[str, Any]], entry: dict[str, Any] | None,
            prev: dict[str, dict[str, Any]], out_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Write the change file (and a snapshot when due); the new manifest entry and the files it no longer needs."""
    for rec in records.values():
        rec["h"] = fmt.record_hash(rec)
    puts = [r for rid, r in records.items() if prev.get(rid, {}).get("h") != r["h"]]
    dels = [{"op": "del", "type": r.get("type"), "id": rid} for rid, r in prev.items() if rid not in records]
    if not puts and not dels:
        return None, []
    if prev and len(dels) > MAX_DELETE_SHARE * len(prev):
        raise RuntimeError(f"{name}: the build would delete {len(dels)} of {len(prev)} records — a source is "
                           "probably broken; nothing published")
    version = (entry or {}).get("version", 0) + 1
    out_dir.mkdir(parents=True, exist_ok=True)
    changes = list((entry or {}).get("changes") or [])
    if entry:
        ch = fmt.write_lines(out_dir / fmt.file_name(name, "change", version),
                             [{"op": "put", **r} for r in puts] + dels)
        changes.append({"version": version, **ch, "puts": len(puts), "dels": len(dels)})
    snap = (entry or {}).get("snapshot")
    since = sum(c["bytes"] for c in changes if snap and c["version"] > snap["version"])
    if not snap or version - snap["version"] >= SNAPSHOT_EVERY or since > snap["bytes"] / 2:
        snap = {"version": version, **fmt.write_lines(out_dir / fmt.file_name(name, "snapshot", version),
                                                       ({"op": "put", **records[k]} for k in sorted(records)))}
    changes = changes[-KEEP_CHANGES:]
    new = {"title": spec.get("title"), "description": " ".join((spec.get("description") or "").split()),
           "version": version, "items": len(records),
           "docs": sum(1 for r in records.values() if r.get("type") == "doc"),
           "recipes": sum(1 for r in records.values() if r.get("type") == "recipe"),
           "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "snapshot": snap, "changes": changes}
    files = lambda e: {e["snapshot"]["file"], *(c["file"] for c in e.get("changes") or [])} if e else set()
    return new, sorted(files(entry) - files(new))


def build(names: list[str], out_dir: Path, base: str | None = None) -> dict[str, Any]:
    """Build the packs into out_dir against the published state at `base` (the release, or a local folder).
    Writes manifest.json there and obsolete.txt with the published files that can be deleted."""
    base = base or fmt.base_url()
    manifest = fmt.load_manifest(base)
    report: dict[str, Any] = {}
    obsolete: list[str] = []
    for name in names:
        spec = load_def(name)
        entry = manifest["packs"].get(name)
        prev = fmt.replay(base, entry)
        records, failed = collect(spec, prev)
        new, gone = publish(name, spec, records, entry, prev, out_dir)
        if new:
            manifest["packs"][name] = new
            obsolete += gone
        report[name] = {"version": (new or entry or {}).get("version"), "changed": bool(new),
                        "items": len(records), "failed_sources": sorted(failed),
                        **({"change": new["changes"][-1] if new["changes"] and
                            new["changes"][-1]["version"] == new["version"] else None,
                            "snapshot": new["snapshot"]} if new else {})}
    manifest.update(format=fmt.FORMAT, updated=datetime.now(timezone.utc).isoformat(timespec="minutes"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / fmt.MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "obsolete.txt").write_text("".join(f + "\n" for f in obsolete), encoding="utf-8")
    if Path(base.removeprefix("file://")).expanduser().resolve() == out_dir.resolve():
        for f in obsolete:   # building into the published folder itself
            (out_dir / f).unlink(missing_ok=True)
    return report
