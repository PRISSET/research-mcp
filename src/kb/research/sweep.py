"""research_sweep: one call that asks every platform at once — the starting point for planning and model choice.

Each platform runs in its own thread with its own database connection; a platform that fails or is rate-limited
is reported, never fatal.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from kb.store import Database

PLATFORMS = ("papers", "models", "repos", "issues", "web", "forums")


def research_sweep(query: str, platforms: list[str] | None = None, per_platform: int = 6) -> dict[str, Any]:
    from kb.research.find import find_papers
    from kb.research.platforms import forum_search, github_search, hf_search
    from kb.research.web import web_search

    use = [p for p in (platforms or PLATFORMS) if p in PLATFORMS] or list(PLATFORMS)
    n = max(3, min(per_platform, 15))
    jobs = {
        "papers": lambda db: find_papers(db, query, n, use=("openalex", "arxiv", "s2", "local")),
        "models": lambda db: hf_search(db, query, "models", "trending", n),
        "repos": lambda db: github_search(db, query, "repos", limit=n),
        "issues": lambda db: github_search(db, query, "issues", limit=n),
        "web": lambda db: web_search(db, query, n),
        "forums": lambda db: forum_search(db, query, limit=max(3, n // 2)),
    }

    def run(name: str) -> Any:
        try:
            db = Database()
        except Exception as e:
            return {"error": f"database: {type(e).__name__}: {str(e)[:160]}"}
        try:
            return jobs[name](db)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)[:160]}"}
        finally:
            db.close()

    Database().close()  # create / migrate the schema once, before the threads race for it
    with ThreadPoolExecutor(max_workers=len(use)) as pool:
        results = dict(zip(use, pool.map(run, use)))

    def slim(v: Any) -> Any:
        if isinstance(v, dict) and "results" in v:
            return v["results"]
        return v

    out = {k: slim(v) for k, v in results.items()}
    if "papers" in out and isinstance(results["papers"], dict):
        out["papers_sources"] = results["papers"].get("sources")
    return {
        "query": query, **out,
        "how_to_use": (
            "Cross-check: official model cards / repos (grade A) for facts about a model; GitHub issues for real-world "
            "problems and fixes; papers for methods and measured numbers; web/forums for recent news and experience. "
            "Read the promising ones (fetch_url / fetch_fulltext). Before recommending a model run latest_models "
            "for the task and model_variants for the user's GPU; check_versions on libraries; and use your own web "
            "search for anything announced in the last weeks."),
    }
