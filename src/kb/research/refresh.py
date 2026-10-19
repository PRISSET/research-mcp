"""notebook_refresh: what appeared since a RESEARCH.md section was last updated, so research does not go stale.

Re-runs the section's saved queries restricted to the period after its date (papers, Hugging Face models,
GitHub repos and issues, Reddit threads with the user's Redlib) and checks the models / repos / packages already cited there for newer versions.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Any

from kb import notebook
from kb.store import Database

STALE_DAYS = 30


def refresh(section: str, since: str | None = None, max_queries: int = 3) -> dict[str, Any]:
    from kb.research.find import find_papers
    from kb.research.platforms import github_search, hf_search
    from kb.research.reddit import reddit_search
    from kb.research.versions import check_one

    sec = notebook.read_section(section)
    start = since or sec.get("updated") or (date.today() - timedelta(days=90)).isoformat()
    from kb.research.digest import topic_query

    # the APIs search in English: a topic without saved English queries is searched by the model it is about
    queries = ([q for q in sec.get("queries") or [] if q.isascii()] or [topic_query(sec)[0] or sec["title"]])[:max_queries]
    have = set(sec["papers"])
    Database().close()   # create / migrate once before the threads

    def run(fn: Any, *args: Any, **kw: Any) -> Any:
        db = Database()
        try:
            return fn(db, *args, **kw)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        finally:
            db.close()

    def newer(items: list[dict[str, Any]], date_key: str = "date") -> list[dict[str, Any]]:
        return [i for i in items if i.get("id") not in have and str(i.get(date_key) or i.get("created") or "") >= start]

    jobs: list[tuple[str, str, Any]] = []
    year = int(start[:4])
    recent = (date.today() - timedelta(days=28)).isoformat()
    for q in queries:
        jobs += [
            ("papers", q, lambda q=q: run(find_papers, q, 8, year)),
            ("models", q, lambda q=q: run(hf_search, q, "models", "created", 8)),
            ("repos", q, lambda q=q: run(github_search, f"{q} created:>={start}", "repos", limit=6)),
            ("issues", q, lambda q=q: run(github_search, f"{q} created:>={start}", "issues", limit=6)),
        ]
        if os.environ.get("REDLIB_URL"):   # practitioners' threads: speed-ups, settings, breakages
            jobs.append(("reddit", q, lambda q=q: run(reddit_search, q, 8, period="year" if start < recent else "month")))
    with ThreadPoolExecutor(max_workers=8) as pool:
        done = [(kind, q, pool.submit(fn)) for kind, q, fn in jobs]
        found: dict[str, dict[str, dict[str, Any]]] = {"papers": {}, "models": {}, "repos": {}, "issues": {}, "reddit": {}}
        for kind, q, fut in done:
            res = fut.result()
            items = res.get("results", []) if isinstance(res, dict) else []
            if kind == "papers":
                db = Database()
                try:
                    items = [{**i, "date": (db.get_paper(i["id"]) or {}).get("published") or str(i.get("year") or "")}
                             for i in items]
                finally:
                    db.close()
            for i in newer(items):
                found[kind].setdefault(i["id"], {**i, "query": q})

        db = Database()
        try:   # ids are lower-cased; the APIs want the original "Org/Name"
            cited = [("gh:" if pid.startswith("gh:") else "hf:", (db.get_paper(pid) or {}).get("title") or pid[3:])
                     for pid in sec["papers"] if pid.startswith(("hf:", "gh:")) and "#" not in pid]
        finally:
            db.close()
        checks = [pool.submit(check_one, f"gh:{name}" if prefix == "gh:" else name) for prefix, name in cited[:10]]
        tracked = [c.result() for c in checks]

    updates = [t for t in tracked if str(t.get("status", "")).startswith("outdated") or t.get("newer_from_same_org")
               or t.get("newer_alternatives") or str(t.get("latest_date") or "") >= start]
    return {
        "section": sec["id"], "since": start, "queries": queries,
        "new_since": {k: list(v.values())[:10] for k, v in found.items()},
        "updates_to_cited": updates,
        "next": ("Read what matters (fetch(url=…) / fetch(paper_ids=…)), add it with notebook_add_source, revise the conclusions "
                 "with notebook_write(summary=…) — that also moves the section's date forward."),
    }


def stale_sections() -> list[dict[str, Any]]:
    today = date.today()
    out = []
    for s in notebook.overview()["sections"]:
        try:
            days = (today - date.fromisoformat(s["updated"])).days
        except (TypeError, ValueError):
            continue
        if days >= STALE_DAYS:
            out.append({"section": s["id"], "updated": s["updated"], "days": days})
    return out
