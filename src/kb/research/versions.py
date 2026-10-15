"""check_versions and diagnose_error: keep the agent on current models and libraries, and look errors up instead
of guessing.

check_versions: PyPI packages ("vllm==0.8.1"), Hugging Face models ("Qwen/Qwen3-Omni-30B-A3B-Instruct": newer
releases of the same family from the same organisation) and GitHub repos ("gh:vllm-project/vllm@v0.9.0").
diagnose_error: the core of an error message, searched verbatim in GitHub issues and PRs, Stack Overflow and the
web at once; closed issues and merged PRs come first because they usually name the fixing version.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from kb.research.sources import interactive_client
from kb.store import Database

GENERIC = {"instruct", "chat", "base", "thinking", "captioner", "fp8", "fp16", "bf16", "awq", "gptq", "gguf", "int4",
           "int8", "hf", "it", "preview", "diffusers", "comfyui", "lora", "distill", "turbo"}


def _vkey(v: str) -> tuple:
    parts = re.findall(r"\d+|[a-z]+", v.lower())
    return tuple((0, int(p)) if p.isdigit() else (-1, p) for p in parts)


def _pypi(name: str, yours: str | None) -> dict[str, Any]:
    http = interactive_client(0.0)
    try:
        data = http.get(f"https://pypi.org/pypi/{name}/json").json()
    finally:
        http.close()
    info = data["info"]
    dated = sorted(((files[0]["upload_time"][:10], v) for v, files in data["releases"].items()
                    if files and not files[0].get("yanked") and not re.search(r"(a|b|rc|dev)\d*$", v)),
                   key=lambda x: _vkey(x[1]))
    latest_date, latest = dated[-1] if dated else (None, info["version"])
    out: dict[str, Any] = {"type": "pypi", "name": name, "latest": latest, "latest_date": latest_date,
                           "links": {k: v for k, v in (info.get("project_urls") or {}).items()
                                     if re.search(r"change|release|source|home|github", k, re.I)}}
    if yours:
        newer = [v for _, v in dated if _vkey(v) > _vkey(yours)]
        out.update(yours=yours, behind=len(newer), newer=newer[-8:],
                   status="up to date" if not newer else f"outdated: {len(newer)} newer releases")
        if yours not in data["releases"]:
            out["status"] = f"version {yours} not on PyPI"
    else:
        out["status"] = "your version unknown — pass name==version"
    return out


def _family(name: str) -> str:
    tokens = [t for t in re.split(r"[-_.]", name) if t]
    words = [t for t in tokens if re.fullmatch(r"[A-Za-z]{3,}", t) and t.lower() not in GENERIC]
    return max(words, key=len) if words else re.sub(r"\d.*$", "", tokens[0]) if tokens else name


def _hf_model(repo_id: str) -> dict[str, Any]:
    http = interactive_client(0.0)
    try:
        meta = http.get(f"https://huggingface.co/api/models/{repo_id}").json()
        org, name = repo_id.split("/", 1)
        fam = _family(name)
        sibs = http.get("https://huggingface.co/api/models",
                        params={"author": org, "search": fam, "sort": "createdAt", "direction": -1, "limit": 30}).json()
    finally:
        http.close()
    created = (meta.get("createdAt") or "")[:10]
    newer = [{"id": m["id"], "created": (m.get("createdAt") or "")[:10]} for m in sibs
             if (m.get("createdAt") or "")[:10] > created and m["id"] != repo_id]
    alternatives = []
    task = meta.get("pipeline_tag")
    if task:
        try:   # newer base models for the same task from anyone, not only this org
            from kb.research.models import latest_models

            lm = latest_models(task, months=12, limit=8)
            alternatives = [{"id": r["id"], "created": r["created"], "likes": r["likes"]}
                            for r in lm["new_base_models"] if r["created"] > created and r["id"] != repo_id][:5]
        except Exception:
            pass
    return {"type": "huggingface", "name": repo_id, "created": created, "updated": (meta.get("lastModified") or "")[:10],
            "task": task, "family": fam, "newer_from_same_org": newer[:10], "newer_alternatives": alternatives,
            "status": (f"{len(newer)} newer '{fam}' models from {org} — check whether one supersedes it" if newer
                       else f"no newer '{fam}' model from {org} on Hugging Face"),
            "note": "Also check the org's GitHub and blog: some releases (API-only versions, renamed families) are not on HF."}


def _github(repo: str, yours: str | None) -> dict[str, Any]:
    import os

    token = os.environ.get("GITHUB_TOKEN")
    http = interactive_client(0.0, {"Authorization": f"Bearer {token}"} if token else None)
    try:
        rel = http.get(f"https://api.github.com/repos/{repo}/releases", params={"per_page": 20}).json()
    finally:
        http.close()
    if not isinstance(rel, list) or not rel:
        return {"type": "github", "name": repo, "status": "no GitHub releases (check tags / commits)"}
    tags = [(r.get("tag_name") or "", (r.get("published_at") or "")[:10]) for r in rel if not r.get("prerelease")]
    out: dict[str, Any] = {"type": "github", "name": repo, "latest": tags[0][0] if tags else None,
                           "latest_date": tags[0][1] if tags else None, "recent": tags[:5]}
    if yours:
        names = [t for t, _ in tags]
        clean = yours.lstrip("v")
        idx = next((i for i, t in enumerate(names) if t.lstrip("v") == clean), None)
        out.update(yours=yours, status="up to date" if idx == 0 else
                   (f"outdated: {idx} newer releases" if idx is not None else "your tag is not among the last 20 releases"))
    return out


BREAKING = re.compile(r"breaking|deprecat|\bremov|drop(?:ped)? support|renam|incompatib|migrat|no longer|"
                      r"minimum .*version|requires? (?:python|torch|cuda)", re.I)
GH_URL = re.compile(r"github\.com/([\w.-]+/[\w.-]+)")


def changelog(repo: str, yours: str, latest: str | None, focus: list[str] | None = None) -> dict[str, Any]:
    """Release notes between `yours` (exclusive) and `latest` (inclusive): breaking changes and lines about `focus`."""
    token = os.environ.get("GITHUB_TOKEN")
    http = interactive_client(0.0, {"Authorization": f"Bearer {token}"} if token else None)
    try:
        rel = http.get(f"https://api.github.com/repos/{repo}/releases", params={"per_page": 100}).json()
    finally:
        http.close()
    if not isinstance(rel, list):
        return {"repo": repo, "error": "no release notes on GitHub"}
    lo, hi = _vkey(yours.lstrip("v")), _vkey((latest or "999").lstrip("v"))
    between = [r for r in rel if not r.get("prerelease") and lo < _vkey((r.get("tag_name") or "").lstrip("v")) <= hi]
    terms = [t.lower() for t in (focus or []) if len(t) > 2]
    breaking, relevant = [], []
    for r in between:
        tag = r.get("tag_name")
        under_breaking = False
        for line in (r.get("body") or "").splitlines():
            if re.match(r"\s*#{1,6}\s|\s*\*\*[^*]+\*\*\s*$", line):   # a heading: are we in a breaking section?
                under_breaking = bool(BREAKING.search(line))
                continue
            text = re.sub(r"\s+", " ", re.sub(r"[*`#>]|\[([^\]]*)\]\([^)]*\)", r"\1", line)).strip(" -")
            if len(text) < 12:
                continue
            if under_breaking or BREAKING.search(text):
                breaking.append({"release": tag, "line": text[:220]})
            if terms and sum(t in text.lower() for t in terms) >= min(2, len(terms)):
                relevant.append({"release": tag, "line": text[:220]})
    return {"repo": repo, "releases_between": [r.get("tag_name") for r in between][:40],
            "breaking": breaking[:15], "relevant_to_focus": relevant[:15],
            "notes": f"https://github.com/{repo}/releases"}


def check_one(item: str, focus: list[str] | None = None) -> dict[str, Any]:
    s = item.strip()
    try:
        if s.startswith(("gh:", "github:", "https://github.com/", "github.com/")):
            repo = re.sub(r"^(gh:|github:|https://github\.com/|github\.com/)", "", s)
            repo, _, ver = repo.partition("@")
            out = {"item": item, **_github(repo.strip("/"), ver or None)}
            if str(out.get("status", "")).startswith("outdated") and ver:
                out["changelog"] = changelog(repo.strip("/"), ver, out.get("latest"), focus)
            return out
        if s.startswith(("hf:", "https://huggingface.co/", "huggingface.co/")) or (
                "/" in s and not re.search(r"[=<>@ ]", s)):
            rid = re.sub(r"^(hf:|https://huggingface\.co/|huggingface\.co/)", "", s).strip("/")
            return {"item": item, **_hf_model(rid)}
        m = re.match(r"^([A-Za-z0-9_.\-\[\]]+?)\s*(?:==|>=|<=|~=|@|\s)\s*v?([\w.\-+]+)$", s)
        name, ver = (re.sub(r"\[.*\]", "", m.group(1)), m.group(2)) if m else (s, None)
        out = {"item": item, **_pypi(name, ver)}
        if str(out.get("status", "")).startswith("outdated") and ver:
            gh = next((GH_URL.search(u).group(1) for u in out.get("links", {}).values() if GH_URL.search(u)), None)
            if gh:
                out["changelog"] = changelog(gh.removesuffix(".git"), ver, out.get("latest"), focus)
        return out
    except Exception as e:
        return {"item": item, "status": f"lookup failed: {type(e).__name__}: {str(e)[:150]}"}


def check_versions(items: list[str], focus: str | None = None) -> dict[str, Any]:
    terms = re.findall(r"[\w][\w.\-]{2,}", focus or "")
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(items)))) as pool:
        results = list(pool.map(lambda i: check_one(i, terms), items[:20]))
    outdated = [r["item"] for r in results
                if str(r.get("status", "")).startswith("outdated") or r.get("newer_from_same_org")
                or r.get("newer_alternatives")]
    return {"results": results, "attention": outdated,
            "next": "For outdated items read the release notes (fetch_url on the changelog / github_read releases) "
                    "before upgrading; a newer model is not automatically better for the user's constraints."}


# ---------------- errors ----------------

NOISE = [
    (re.compile(r"(?:/[\w.\-]+)+\.(?:py|so|cpp|cu|h)(?::\d+)?"), " "),           # file paths
    (re.compile(r"0x[0-9a-fA-F]+"), " "),                                      # addresses
    (re.compile(r"\b\d{3,}\b"), " "),                                          # sizes, line numbers, pids
    (re.compile(r"'[^']{40,}'|\"[^\"]{40,}\""), " "),                          # long quoted values
]


def error_core(error: str) -> str:
    """The searchable part of an error: the last 'XError: message' line, minus paths, numbers and long values."""
    lines = [ln.strip() for ln in error.strip().splitlines() if ln.strip()]
    pick = next((ln for ln in reversed(lines) if re.search(r"(Error|Exception|error:|failed|Failed|FATAL|panic)", ln)),
                lines[-1] if lines else error)
    pick = re.sub(r"^.*?\b([A-Z]\w*(?:Error|Exception))\s*:\s*", r"\1: ", pick)
    for pat, rep in NOISE:
        pick = pat.sub(rep, pick)
    return " ".join(pick.split())[:200]


TRAILING = {"for", "on", "with", "in", "at", "of", "to", "the", "a", "an", "request", "id", "from", "by", "is", "was"}


def _phrase(core: str, words: int = 12) -> str:
    msg = core.split(": ", 1)[-1]
    toks = re.sub(r"[\"`\[\](){}]", " ", msg).split()[:words]
    while toks and toks[-1].lower() in TRAILING:     # "... embedding for request <id>" -> "... embedding"
        toks.pop()
    return " ".join(toks)


def phrase_candidates(core: str) -> list[str]:
    """Exact message first, then shorter heads: GitHub's phrase search needs a verbatim match."""
    out: list[str] = []
    for n in (12, 8, 5):
        p = _phrase(core, n)
        if p and p not in out and len(p.split()) >= 2:
            out.append(p)
    return out


def diagnose_error(db_factory: Any, error: str, library: str | None = None, version: str | None = None,
                   context: str | None = None) -> dict[str, Any]:
    from kb.research.platforms import forum_search, github_search
    from kb.research.web import web_search

    core = error_core(error)
    candidates = phrase_candidates(core) or [core]
    phrase = candidates[0]

    db_factory().close()  # create / migrate the schema once, before the threads race for it

    def run(fn: Any, *args: Any, **kw: Any) -> Any:
        try:
            db: Database = db_factory()
        except Exception as e:
            return {"error": f"database: {type(e).__name__}: {str(e)[:150]}"}
        try:
            return fn(db, *args, **kw)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        finally:
            db.close()

    def github() -> tuple[str, list[dict], list[dict]]:
        # One phrase at a time (anonymous search allows 10 a minute): the first that matches wins.
        for cand in candidates:
            got = run(github_search, f'"{cand}"', kind="issues", limit=8)
            issues = got.get("results", []) if isinstance(got, dict) else []
            if issues:
                prs = run(github_search, f'"{cand}"', kind="prs", limit=5)
                return cand, issues, prs.get("results", []) if isinstance(prs, dict) else []
        return candidates[-1], [], []

    def web() -> Any:
        for q in (f'"{phrase}" {library or ""}', f"{phrase} {library or ''}"):
            got = run(web_search, q.strip(), limit=6)
            if isinstance(got, dict) and got.get("results"):
                return got["results"]
        return got

    with ThreadPoolExecutor(max_workers=4) as pool:
        gh = pool.submit(github)
        so = pool.submit(run, forum_search, phrase, sites=["stackoverflow"], limit=5)
        wb = pool.submit(web)
        focus_terms = [w for w in phrase.split() if len(w) > 3][:6]
        ver = pool.submit(check_one, f"{library}=={version}" if version else library, focus_terms) if library else None
        matched, issues, prs = gh.result()
        so_res, web_res = so.result(), wb.result()
        version_check = ver.result() if ver else None

    issues.sort(key=lambda i: (i.get("state") != "closed", -(i.get("comments") or 0)))
    merged = [p for p in prs if p.get("merged")]
    hints = []
    if merged:
        hints.append(f"{len(merged)} merged PR(s) mention this error — a fix likely exists; find the release that ships it")
    if any(i.get("state") == "closed" for i in issues):
        hints.append("closed issues match — read them (github_read) for the fix or workaround and the version")
    if issues and all(i.get("state") == "open" for i in issues):
        hints.append("only open issues match — likely a known, unfixed bug: look for workarounds in the comments")
    if not issues and not prs:
        hints.append("nothing on GitHub for the exact message — try your own web search with the error text, "
                     "and check the docs / changelog of the library")
    if version_check and str(version_check.get("status", "")).startswith("outdated"):
        hints.insert(0, f"you are on an outdated {library} ({version_check['status']}) — the error may be fixed upstream")
        rel = (version_check.get("changelog") or {}).get("relevant_to_focus") or []
        if rel:
            hints.insert(0, f"release notes after your version mention this: {rel[0]['release']}: {rel[0]['line']}")
    return {"searched_for": core, "matched_phrase": matched, "hints": hints, "version_check": version_check,
            "github_issues": issues[:8], "github_prs": prs[:5],
            "stackoverflow": so_res.get("stackoverflow") if isinstance(so_res, dict) else so_res, "web": web_res,
            "context": context,
            "next": "github_read(repo, issue=N) on the best match, then apply the documented fix; record what worked "
                    "in RESEARCH.md if it took effort to find."}
