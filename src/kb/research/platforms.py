"""GitHub, Hugging Face, Hacker News and Stack Overflow — practical knowledge that papers lack: working configs,
known bugs, model cards, release notes, community experience. No keys needed.

GitHub without a token: 10 searches a minute and 60 other API calls an hour (READMEs come from
raw.githubusercontent.com and do not count). GITHUB_TOKEN, if the user sets it, raises the limits.
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import date
from typing import Any

from kb.critic.sources import grade_doc
from kb.research.docs import Doc, index_text, upsert_doc
from kb.research.sources import interactive_client
from kb.store import Database

GH_API = "https://api.github.com"
HF = "https://huggingface.co"


def _gh() -> Any:
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", **({"Authorization": f"Bearer {token}"} if token else {})}
    return interactive_client(0.3, headers)


def _date(s: str | None) -> str | None:
    return s[:10] if s else None


def _save(db: Database, d: Doc) -> dict[str, Any]:
    upsert_doc(db, d)
    g = grade_doc(d.kind, d.extra, d.id)
    db.conn.execute("INSERT OR REPLACE INTO critique (source_id, kind, grade, reasons, flags) VALUES (?, ?, ?, ?, ?)",
                    (d.id, d.kind, g["grade"], json.dumps(g["reasons"]), json.dumps(g["flags"])))
    return {"id": d.id, "kind": d.kind, "title": d.title, "url": d.url, "summary": d.summary[:400],
            "date": d.published, **{k: v for k, v in d.extra.items() if k in SHOWN}, "grade": g["grade"], "flags": g["flags"]}


SHOWN = {"created", "age_months", "relation", "stars", "forks", "pushed", "archived", "license", "language", "state", "merged", "comments", "repo",
         "downloads", "likes", "pipeline", "library", "params", "gated", "base_model", "points", "score", "answered",
         "updated", "author"}


# ---------------- GitHub ----------------

def repo_doc(r: dict[str, Any]) -> Doc:
    return Doc(
        id=f"gh:{r['full_name'].lower()}", kind="repo", title=r["full_name"], url=r["html_url"], platform="github",
        summary=r.get("description") or "", published=_date(r.get("created_at")), authors=[r["owner"]["login"]],
        extra={"stars": r.get("stargazers_count"), "forks": r.get("forks_count"), "pushed": _date(r.get("pushed_at")),
               "archived": r.get("archived"), "fork": r.get("fork"), "license": (r.get("license") or {}).get("spdx_id"),
               "language": r.get("language"), "topics": r.get("topics") or [], "open_issues": r.get("open_issues_count"),
               "owner": r["owner"]["login"]},
    )


def issue_doc(i: dict[str, Any]) -> Doc:
    repo = "/".join(i["repository_url"].split("/")[-2:])
    is_pr = "pull_request" in i
    return Doc(
        id=f"gh:{repo.lower()}#{i['number']}", kind="pr" if is_pr else "issue", title=f"{repo}#{i['number']}: {i['title']}",
        url=i["html_url"], platform="github", summary=(i.get("body") or "")[:1500], published=_date(i.get("created_at")),
        authors=[(i.get("user") or {}).get("login", "")],
        extra={"repo": repo, "state": i.get("state"), "comments": i.get("comments"), "updated": _date(i.get("updated_at")),
               "merged": bool((i.get("pull_request") or {}).get("merged_at")) if is_pr else None,
               "labels": [lb["name"] for lb in i.get("labels") or []],
               "reactions": (i.get("reactions") or {}).get("total_count"), "owner": repo.split("/")[0]},
    )


def github_search(db: Database, query: str, kind: str = "repos", repo: str | None = None, limit: int = 10,
                  state: str | None = None) -> dict[str, Any]:
    http = _gh()
    try:
        if kind == "repos":
            data = http.get(f"{GH_API}/search/repositories", params={"q": f"{query} in:name,description,readme",
                                                                     "per_page": min(limit, 50)}).json()
            out = [_save(db, repo_doc(r)) for r in data.get("items", [])]
        else:
            q = f"{query} {'is:pr' if kind == 'prs' else 'is:issue'}"
            if repo:
                q += f" repo:{repo}"
            if state in ("open", "closed"):
                q += f" state:{state}"
            data = http.get(f"{GH_API}/search/issues", params={"q": q, "per_page": min(limit, 50)}).json()
            out = [_save(db, issue_doc(i)) for i in data.get("items", [])]
        db.commit()
        return {"query": query, "kind": kind, "total": data.get("total_count"), "results": out}
    finally:
        http.close()


def _raw_readme(http: Any, repo: str) -> str:
    for name in ("README.md", "readme.md", "README.rst", "README"):
        try:
            r = http.get(f"https://raw.githubusercontent.com/{repo}/HEAD/{name}")
            if r.status_code == 200 and r.text.strip():
                return r.text
        except Exception:
            continue
    return ""


def github_read(db: Database, repo: str, issue: int | None = None, path: str | None = None,
                releases: bool = True) -> dict[str, Any]:
    """A repo (metadata + README + recent release notes), one file, or an issue / PR thread with all comments."""
    repo = repo.strip().removeprefix("https://github.com/").strip("/")
    http = _gh()
    try:
        if issue is not None:
            i = http.get(f"{GH_API}/repos/{repo}/issues/{issue}").json()
            comments = http.get(f"{GH_API}/repos/{repo}/issues/{issue}/comments", params={"per_page": 100}).json()
            d = issue_doc({**i, "repository_url": f"{GH_API}/repos/{repo}"})
            parts = [f"# {i['title']}", f"State: {i.get('state')} · labels: {', '.join(d.extra['labels']) or '-'}",
                     f"## Opening post by {d.authors[0]} ({d.published})", i.get("body") or ""]
            for c in comments if isinstance(comments, list) else []:
                role = c.get("author_association", "")
                tag = " (maintainer)" if role in ("OWNER", "MEMBER", "COLLABORATOR") else ""
                parts += [f"## Comment by {(c.get('user') or {}).get('login')}{tag} ({_date(c.get('created_at'))})",
                          c.get("body") or ""]
            d.extra["maintainer_replied"] = any((c.get("author_association") in ("OWNER", "MEMBER", "COLLABORATOR"))
                                                for c in comments if isinstance(c, dict))
            text = "\n\n".join(parts)
        elif path:
            r = http.get(f"https://raw.githubusercontent.com/{repo}/HEAD/{path.lstrip('/')}")
            d = Doc(id=f"gh:{repo.lower()}/{path.lstrip('/')}", kind="web", title=f"{repo}/{path}", platform="github",
                    url=f"https://github.com/{repo}/blob/HEAD/{path.lstrip('/')}", extra={"repo": repo, "owner": repo.split("/")[0]})
            text = r.text if path.endswith((".md", ".rst", ".txt")) else f"```\n{r.text}\n```"
        else:
            meta = http.get(f"{GH_API}/repos/{repo}").json()
            d = repo_doc(meta)
            parts = [_raw_readme(http, repo)]
            if releases:
                try:
                    rel = http.get(f"{GH_API}/repos/{repo}/releases", params={"per_page": 5}).json()
                    for r in rel if isinstance(rel, list) else []:
                        parts.append(f"## Release {r.get('tag_name')} ({_date(r.get('published_at'))})\n\n{r.get('body') or ''}")
                    if isinstance(rel, list) and rel:
                        d.extra["latest_release"] = rel[0].get("tag_name")
                        d.extra["latest_release_date"] = _date(rel[0].get("published_at"))
                except Exception:
                    pass
            text = "\n\n".join(p for p in parts if p)
        brief = _save(db, d)
        idx = index_text(db, d.id, text, "github", d.title)
        db.commit()
        return {**brief, "indexed": idx, "text": text[:12000], "truncated": len(text) > 12000,
                "outline": _outline(text)}
    finally:
        http.close()


def _outline(md: str) -> list[str]:
    return [m.group(0)[:100] for m in re.finditer(r"^#{1,3} .+$", md, re.M)][:60]


# ---------------- Hugging Face ----------------

HF_SORT = {"downloads": "downloads", "likes": "likes", "trending": "trendingScore", "recent": "lastModified",
           "created": "createdAt"}


def _hf_doc(item: dict[str, Any], kind: str) -> Doc:
    rid = item.get("id") or item.get("modelId")
    tags = item.get("tags") or []
    prefix = {"model": "hf:", "dataset": "hf-dataset:", "space": "hf-space:"}[kind]
    url = f"{HF}/{'datasets/' if kind == 'dataset' else 'spaces/' if kind == 'space' else ''}{rid}"
    card = item.get("cardData") or {}
    lic = card.get("license") or next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
    base = [t.split(":", 2)[-1] for t in tags if t.startswith("base_model:") and t.count(":") >= 2]
    extra = {"downloads": item.get("downloads"), "likes": item.get("likes"), "license": lic,
             "pipeline": item.get("pipeline_tag"), "library": item.get("library_name"), "gated": item.get("gated"),
             "base_model": sorted(set(base))[:5], "updated": _date(item.get("lastModified")),
             "created": _date(item.get("createdAt")), "owner": rid.split("/")[0] if "/" in rid else rid}
    if extra["created"]:
        c = date.fromisoformat(extra["created"])
        extra["age_months"] = (date.today().year - c.year) * 12 + date.today().month - c.month
    rel = next((t.split(":")[1] for t in tags if t.startswith("base_model:") and t.count(":") >= 2), None)
    if rel:
        extra["relation"] = rel          # quantized | finetune | adapter | merge of base_model
    st = item.get("safetensors") or {}
    if st.get("total"):
        extra["params"] = st["total"]
    files = [s.get("rfilename", "") for s in item.get("siblings") or []]
    if files:
        extra["safetensors"] = any(f.endswith(".safetensors") for f in files)
        extra["pickle_files"] = [f for f in files if f.endswith((".bin", ".pt", ".pth", ".ckpt", ".pkl"))][:5]
        extra["gguf"] = any(f.endswith(".gguf") for f in files)
    return Doc(id=f"{prefix}{rid.lower()}", kind=kind, title=rid, url=url, platform="huggingface",
               summary=card.get("description") or "", published=_date(item.get("createdAt")), authors=[extra["owner"]],
               extra=extra)


def hf_search(db: Database, query: str, kind: str = "models", sort: str = "trending", limit: int = 10) -> dict[str, Any]:
    http = interactive_client(0.2)
    try:
        if kind == "papers":
            data = http.get(f"{HF}/api/papers/search", params={"q": query}).json()
            out = []
            for row in data[:limit]:
                p = row.get("paper") or row
                out.append({"id": f"arxiv:{p.get('id')}", "title": p.get("title"), "upvotes": p.get("upvotes"),
                            "date": _date(p.get("publishedAt")), "summary": (p.get("summary") or "")[:400],
                            "next": "find_papers / fetch(paper_ids=[…]) with this arXiv id"})
            return {"query": query, "kind": kind, "results": out}
        single = kind.rstrip("s")
        seen: dict[str, Any] = {}
        exact: set[str] = set()
        tried = []
        # HF matches a substring of the repo id: a long natural query finds nothing, so fall back to keywords.
        for q in [query, *_hf_keywords(query)]:
            if len(seen) >= limit * 2 or len(tried) >= 4:
                break
            tried.append(q)
            params = {"search": q, "sort": HF_SORT.get(sort, "downloads"), "direction": -1, "limit": min(limit, 50)}
            for item in http.get(f"{HF}/api/{kind}", params=params).json():
                rid = item.get("id") or item.get("modelId")
                seen.setdefault(rid, item)
                if q == query:
                    exact.add(rid)
        keys = [w for w in re.findall(r"[\w][\w.\-]*", query.lower()) if w not in HF_STOP and len(w) > 2]

        def relevance(item: dict[str, Any]) -> int:
            hay = " ".join([(item.get("id") or "").lower(), *(t.lower() for t in item.get("tags") or []),
                            (item.get("pipeline_tag") or "").lower()])
            return sum(1 for k in keys if k in hay or k.split("-")[0] in hay)

        need = min(2, len(keys))
        # Keyword fallbacks are noisy ("recognition" also matches table-recognition): keep items that match
        # at least two words of the request in their id or tags (task, language).
        items = [i for rid, i in seen.items() if rid in exact or relevance(i) >= need]
        if sort == "trending":
            items.sort(key=lambda i: (i.get("id") in exact, min(relevance(i), need), i.get("trendingScore") or 0,
                                      i.get("likes") or 0), reverse=True)
        elif sort in ("downloads", "likes"):  # merged over several keyword queries: re-rank as one list
            items.sort(key=lambda i: (i.get("id") in exact, min(relevance(i), need), i.get(sort) or 0), reverse=True)
        out = [_save(db, _hf_doc(item, single)) for item in items[:limit]]
        db.commit()
        return {"query": query, "kind": kind, "sort": sort, "searched": tried, "results": out}
    finally:
        http.close()


HF_STOP = {"open", "source", "model", "models", "best", "generation", "video", "image", "text", "using", "with", "for",
           "and", "the", "how", "local", "free", "new", "latest", "fast", "small", "large"}


def _hf_keywords(query: str) -> list[str]:
    words = [w for w in re.findall(r"[\w][\w.\-]*", query.lower()) if w not in HF_STOP and len(w) > 2]
    if len(words) <= 1 and query.strip().lower() == (words[0] if words else ""):
        return []
    hyphenated = [w for w in words if "-" in w]
    singles = sorted({w for w in words if "-" not in w}, key=len, reverse=True)
    return list(dict.fromkeys(hyphenated + singles))[:4]


def hf_read(db: Database, repo_id: str, kind: str = "model") -> dict[str, Any]:
    """Model / dataset / space card + metadata (license, gating, parameters, file formats, base models)."""
    rid = re.sub(r"^https?://huggingface\.co/(datasets/|spaces/)?", "", repo_id.strip()).strip("/")
    api = {"model": "models", "dataset": "datasets", "space": "spaces"}[kind]
    http = interactive_client(0.2)
    try:
        meta = http.get(f"{HF}/api/{api}/{rid}").json()
        d = _hf_doc(meta, kind)
        prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}[kind]
        try:
            card = http.get(f"{HF}/{prefix}{rid}/raw/main/README.md").text
        except Exception:
            card = ""
        brief = _save(db, d)
        idx = index_text(db, d.id, card, "huggingface", d.title) if card else {"chunks": 0}
        db.commit()
        return {**brief, "files": [s.get("rfilename") for s in meta.get("siblings") or []][:80],
                "indexed": idx, "card": card[:12000], "truncated": len(card) > 12000, "outline": _outline(card)}
    finally:
        http.close()


# ---------------- Forums ----------------

def forum_search(db: Database, query: str, sites: list[str] | None = None, limit: int = 8) -> dict[str, Any]:
    sites = sites or ["hackernews", "stackoverflow"]
    http = interactive_client(0.2)
    out: dict[str, Any] = {}
    try:
        if "hackernews" in sites:
            try:
                data = http.get("https://hn.algolia.com/api/v1/search",
                                params={"query": query, "tags": "story", "hitsPerPage": limit}).json()
                out["hackernews"] = [_save(db, Doc(
                    id=f"hn:{h['objectID']}", kind="hn", title=h.get("title") or "", platform="hackernews",
                    url=f"https://news.ycombinator.com/item?id={h['objectID']}", published=_date(h.get("created_at")),
                    authors=[h.get("author") or ""],
                    extra={"points": h.get("points"), "comments": h.get("num_comments"), "link": h.get("url")}))
                    for h in data.get("hits", [])]
            except Exception as e:
                out["hackernews"] = {"error": f"{type(e).__name__}: {e}"[:200]}
        if "stackoverflow" in sites:
            try:
                data = http.get("https://api.stackexchange.com/2.3/search/advanced",
                                params={"order": "desc", "sort": "relevance", "q": query, "site": "stackoverflow",
                                        "pagesize": limit}).json()
                out["stackoverflow"] = [_save(db, Doc(
                    id=f"so:{q['question_id']}", kind="so", title=_unescape(q.get("title", "")), platform="stackoverflow",
                    url=q.get("link", ""), published=date.fromtimestamp(q["creation_date"]).isoformat(),
                    extra={"score": q.get("score"), "answered": bool(q.get("accepted_answer_id")) or q.get("is_answered"),
                           "answers": q.get("answer_count"), "tags": q.get("tags", [])}))
                    for q in data.get("items", [])]
            except Exception as e:
                out["stackoverflow"] = {"error": f"{type(e).__name__}: {e}"[:200]}
        db.commit()
        return {"query": query, **out, "next": "fetch(url=…) to read a thread in full"}
    finally:
        http.close()


def _unescape(s: str) -> str:
    return html.unescape(s)
