"""research_sweep: one call that asks every platform at once — the starting point for planning and model choice.

Each platform runs in its own thread with its own database connection; a platform that fails or is rate-limited
is reported, never fatal.

"community" is where practitioners talk: Hugging Face discussions of the matching models, Civitai checkpoints and
LoRAs and ComfyUI nodes (for generative topics), Hacker News, and Reddit through the user's own Redlib. "web" without a search key would only scrape
DuckDuckGo / Bing, which returns noise: the agent gets ready queries for its own web search instead (Reddit
included — its API is closed to new apps).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from kb.store import Database

PLATFORMS = ("papers", "models", "repos", "issues", "community", "videos", "web")
COMMUNITY = ("discussions", "reddit", "civitai", "nodes", "hn")
ALIASES = {"forums": "community"}


def _own_db(fn: Any) -> Any:
    try:
        db = Database()
    except Exception as e:
        return {"error": f"database: {type(e).__name__}: {str(e)[:160]}"}
    try:
        return fn(db)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:160]}"}
    finally:
        db.close()


def community(query: str, parts: set[str], n: int) -> dict[str, Any]:
    from kb.research.platforms import civitai_search, comfy_nodes, forum_search, hf_discussions, hf_search
    from kb.research.reddit import reddit_search

    def discussions(db: Database) -> Any:
        # threads gather on the base models (finetunes and quants rarely have any): the bases the matching models
        # are built on, most common first, then the top match itself
        from collections import Counter

        models = hf_search(db, query, "models", "downloads", 6).get("results") or []
        bases = Counter(b for m in models for b in m.get("base_model") or [])
        ids = [b for b, _ in bases.most_common(2)] + [m["title"] for m in models[:1]]
        return hf_discussions(db, list(dict.fromkeys(ids))[:3], query, limit=3)

    jobs = {
        "discussions": discussions,
        "reddit": lambda db: reddit_search(db, query, max(3, n // 2)),
        "civitai": lambda db: civitai_search(db, query, n),
        "nodes": lambda db: comfy_nodes(db, query, max(3, n // 2)),
        "hn": lambda db: forum_search(db, query, ["hackernews"], limit=3),
    }
    use = [p for p in COMMUNITY if p in parts]
    with ThreadPoolExecutor(max_workers=len(use) or 1) as pool:
        res = dict(zip(use, pool.map(lambda p: _own_db(jobs[p]), use)))
    out: dict[str, Any] = {}
    for k, v in res.items():
        if isinstance(v, dict) and "error" in v:
            out[k] = v
        elif k == "hn":
            out[k] = v.get("hackernews") or []
        else:
            out[k] = v.get("results") or []
    return out


def web_queries(query: str) -> dict[str, Any]:
    reddit = [] if os.environ.get("REDLIB_URL") else [f"{query} site:reddit.com"]   # else community.reddit has it
    return {"use_your_own_search": [query, *reddit, f"{query} review OR comparison OR benchmark"],
            "note": "Built-in web search is off without SearXNG / TAVILY_API_KEY / BRAVE_API_KEY (scraped results are noise). Run "
                    "these in your own web search and pass the useful pages to fetch(urls=[…]) so they are cached "
                    "and citable; platforms=['web'] forces the scraper."}


def research_sweep(query: str, platforms: list[str] | None = None, per_platform: int = 6) -> dict[str, Any]:
    from kb.keys import missing_hint
    from kb.research.find import find_papers
    from kb.research.platforms import GENERATIVE, github_search, hf_search
    from kb.research.web import web_search
    from kb.research.youtube import youtube_search

    asked = [ALIASES.get(p, p) for p in (platforms or [])]
    parts = {p for p in asked if p in COMMUNITY}
    use = [p for p in dict.fromkeys(asked) if p in PLATFORMS] or ([] if parts else list(PLATFORMS))
    if parts and "community" not in use:
        use.append("community")
    if "community" in use and not parts:
        parts = ({"discussions", "hn"} | ({"civitai", "nodes"} if GENERATIVE.search(query) else set())
                 | ({"reddit"} if os.environ.get("REDLIB_URL") else set()))
    from kb.research.web import has_engine

    has_web_key = has_engine()
    forced_web = "web" in asked
    n = max(3, min(per_platform, 15))
    jobs = {
        "papers": lambda: _own_db(lambda db: find_papers(db, query, n)),
        "models": lambda: _own_db(lambda db: hf_search(db, query, "models", "trending", n)),
        "repos": lambda: _own_db(lambda db: github_search(db, query, "repos", limit=n)),
        "issues": lambda: _own_db(lambda db: github_search(db, query, "issues", limit=n)),
        "community": lambda: community(query, parts, n),
        "videos": lambda: _own_db(lambda db: youtube_search(db, query, max(3, n // 2))),
        "web": (lambda: _own_db(lambda db: web_search(db, query, n))) if has_web_key or forced_web
        else (lambda: web_queries(query)),
    }

    Database().close()  # create / migrate the schema once, before the threads race for it
    with ThreadPoolExecutor(max_workers=len(use)) as pool:
        results = dict(zip(use, pool.map(lambda p: jobs[p](), use)))

    def slim(v: Any) -> Any:
        if isinstance(v, dict) and "results" in v:
            return v["results"]
        return v

    out = {k: slim(v) for k, v in results.items()}
    if "papers" in out and isinstance(results["papers"], dict):
        out["papers_sources"] = results["papers"].get("sources")
    hint = missing_hint((["SEARXNG_URL"] if not has_web_key else []) + ["REDLIB_URL", "GITHUB_TOKEN", "S2_API_KEY"])
    return {
        "query": query, **out,
        "how_to_use": (
            "Cross-check: official model cards / repos (grade A) for facts about a model; GitHub issues and HF "
            "discussions for real-world problems and fixes; Civitai and ComfyUI nodes for what the community "
            "actually runs; YouTube for walkthroughs (fetch reads the transcript with timestamps); papers for methods and measured numbers. Read the promising ones (fetch(url=…) / "
            "fetch(paper_ids=…)). Before recommending a model run latest_models for the task and model_variants "
            "for the user's GPU; check_versions on libraries; and use your own web search (Reddit, blogs) for "
            "experience and anything announced in the last weeks."),
        **({"keys": hint} if hint else {}),
    }
