"""News for the user's research topics: what appeared since the last session, checked in the background.

At server start (at most once a day) every recent RESEARCH.md topic is checked for what is new since it was last
checked: trending new Hugging Face models, new papers, Civitai checkpoints / LoRAs and ComfyUI nodes for generative
topics, Reddit threads (with the user's Redlib), newer releases of the models and repos the topic cites. notebook_read() shows the items once and marks
them shown, so the agent can open a session with "since last time: …" for the topics the user works on.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from kb import notebook
from kb.config import settings
from kb.store import Database

log = logging.getLogger(__name__)

EVERY_HOURS = 20
MAX_SECTIONS = 6
MAX_BACK_DAYS = 60
FIRST_LOOK_DAYS = 14
PER_SECTION = 6
DEADLINE = 90.0
_running = threading.Event()


def _path() -> Path:
    return settings.data_dir / "digest.json"


def _load() -> dict[str, Any]:
    try:
        all_ = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return all_.get(str(notebook.notebook_path()), {})


def _save(state: dict[str, Any]) -> None:
    p = _path()
    try:
        all_ = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        all_ = {}
    all_[str(notebook.notebook_path())] = state
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(all_, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def _run(fn: Any, *args: Any, **kw: Any) -> Any:
    db = Database()
    try:
        return fn(db, *args, **kw)
    finally:
        db.close()


GENERIC = {"ai", "gb", "vram", "rtx", "ti", "fps", "nsfw", "lora", "gpu", "top", "vs"}
MODEL_NAME = re.compile(r"(?:\b[A-Za-z][\w.-]*\s+)?\b[A-Za-z]*\d[\w.]*")


def topic_query(sec: dict[str, Any]) -> tuple[str, bool]:
    """An English search for the topic: its saved query; else a model name in its title ("MiniMax H3",
    "Qwen 2.1"); else the model family it cites most (Hugging Face ids count double); else the title's Latin
    words. The flag says whether it is a real research query (worth a paper search)."""
    from collections import Counter

    from kb.research.models import _family_key

    queries = [q for q in sec.get("queries") or [] if q.isascii()]
    if queries:
        return queries[0], True
    title = sec["title"]
    fams: Counter[str] = Counter()
    for pid in sec.get("papers") or []:
        if pid.startswith(("hf:", "gh:")) and (k := _family_key(pid[3:].split("#")[0])):
            fams[k] += 2 if pid.startswith("hf:") else 1
    for m in MODEL_NAME.finditer(title):
        words = [w for w in m.group(0).split() if w.lower() not in GENERIC]
        if words and re.search(r"[A-Za-z]", words[-1]) or len(words) == 2:
            name = " ".join(words)
            longer = [k for k, _ in fams.most_common() if len(name) <= 3 and name.lower() in k.split("-")]
            return (longer[0].replace("-", " ") if longer else name), False   # "H3" -> "minimax h3"
    if fams:
        return fams.most_common(1)[0][0].replace("-", " "), False
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9.]+", title) if w.lower() not in GENERIC]
    return " ".join(words[:3]), False


def _section_news(sec: dict[str, Any], since: str, query: str, research: bool) -> list[dict[str, Any]]:
    """Jobs for one topic, run in parallel; each returns items dated on or after `since`."""
    from kb.research.find import find_papers
    from kb.research.platforms import GENERATIVE, civitai_search, comfy_nodes, hf_search
    from kb.research.versions import check_one

    have = set(sec.get("papers") or [])

    def models() -> list[dict[str, Any]]:
        res = _run(hf_search, query, "models", "trending", 20).get("results") or []
        return [{"kind": "model", "title": m["title"], "url": m["url"], "date": m.get("created") or m.get("date"),
                 "why": f"new model, {m.get('likes') or 0} likes, {m.get('downloads') or 0} downloads"
                        + (f", based on {', '.join(m['base_model'][:1])}" if m.get("base_model") else ""),
                 "_rank": (m.get("likes") or 0) * 10 + (m.get("downloads") or 0) / 100}
                for m in res if (m.get("likes") or 0) + (m.get("downloads") or 0) / 100 >= 2]

    def papers() -> list[dict[str, Any]]:
        res = _run(find_papers, query, 8, int(since[:4])).get("results") or []
        db = Database()
        try:
            out = []
            for p in res:
                d = (db.get_paper(p["id"]) or {}).get("published") or str(p.get("year") or "")
                out.append({"kind": "paper", "title": p.get("title"), "url": p.get("url"), "date": d, "id": p["id"],
                            "why": "new paper" + (f", {p['citations']} citations" if p.get("citations") else ""),
                            "_rank": 5 + (p.get("citations") or 0)})
            return out
        finally:
            db.close()

    def civitai() -> list[dict[str, Any]]:
        res = _run(civitai_search, query, 10, sort="Newest").get("results") or []
        return [{"kind": "civitai", "title": c["title"], "url": c["url"], "date": c.get("date"),
                 "why": f"Civitai {c.get('type') or 'model'}, {c.get('downloads') or 0} downloads",
                 "_rank": (c.get("downloads") or 0) / 50} for c in res if (c.get("downloads") or 0) >= 100]

    def nodes() -> list[dict[str, Any]]:
        res = _run(comfy_nodes, query, 8).get("results") or []
        return [{"kind": "node", "title": n["title"], "url": n["url"], "date": n.get("created") or n.get("date"),
                 "why": f"new ComfyUI node, {n.get('stars') or 0} stars" + (" — flagged" if n.get("grade") == "D" else ""),
                 "_rank": n.get("stars") or 0} for n in res]

    def reddit() -> list[dict[str, Any]]:
        from kb.research.reddit import reddit_search

        res = _run(reddit_search, query, 8, period="month").get("results") or []
        return [{"kind": "reddit", "title": r["title"], "url": r["url"], "date": r.get("date"),
                 "why": f"Reddit thread, {r.get('score') or 0} upvotes, {r.get('comments_n') or 0} comments",
                 "_rank": -n} for n, r in enumerate(res) if (r.get("score") or 0) >= 10]   # Reddit's relevance
                                                                                         # order: upvotes favour showcases

    def cited() -> list[dict[str, Any]]:
        db = Database()
        try:   # ids are lower-cased; the APIs want the original "Org/Name"
            items = [("gh:" if pid.startswith("gh:") else "", (db.get_paper(pid) or {}).get("title") or pid[3:])
                     for pid in have if pid.startswith(("hf:", "gh:")) and "#" not in pid][:5]
        finally:
            db.close()
        out = []
        for prefix, name in items:
            t = check_one(prefix + name)
            if t.get("type") == "github" and str(t.get("latest_date") or "") >= since:
                out.append({"kind": "update", "title": name, "url": f"https://github.com/{name}/releases",
                            "date": str(t["latest_date"]), "why": f"cited here; new release {t.get('latest')}",
                            "_rank": 1000})
            for m in t.get("newer_from_same_org") or []:   # same family, same org, released since the last check
                if m["created"] >= since:
                    out.append({"kind": "update", "title": m["id"], "url": f"https://huggingface.co/{m['id']}",
                                "date": m["created"], "why": f"newer '{t.get('family')}' model than {name} (cited here)",
                                "_rank": 1000})
        return out

    jobs = ([cited] + ([models] if query else []) + ([papers] if research else [])
            + ([reddit] if query and os.environ.get("REDLIB_URL") else []))
    if query and GENERATIVE.search(f"{sec['title']} {query} {' '.join(have)}"):
        jobs += [civitai, nodes]
    pool = ThreadPoolExecutor(max_workers=len(jobs))
    futs = [pool.submit(j) for j in jobs]
    done, _ = wait(futs, timeout=DEADLINE)
    pool.shutdown(wait=False, cancel_futures=True)
    items: list[dict[str, Any]] = []
    for f in futs:
        if f in done and not f.exception():
            items += [i for i in f.result() if str(i.get("date") or "") >= since and i.get("id") not in have]
        elif f in done:
            log.info("digest %s: %s", sec["id"], f.exception())
    items.sort(key=lambda i: i.pop("_rank", 0), reverse=True)
    # the best of each kind in turn: model likes and Reddit upvotes are not on one scale
    by: dict[str, list[dict[str, Any]]] = {}
    for i in items:
        by.setdefault(i["kind"], []).append(i)
    out: list[dict[str, Any]] = []
    while len(out) < PER_SECTION and any(by.values()):
        for kind in list(by):
            if by[kind] and len(out) < PER_SECTION:
                out.append(by[kind].pop(0))
    return out


def build() -> dict[str, Any]:
    """Check the most recently updated topics for news since their last check (or their own date)."""
    state = _load()
    checked = state.get("sections_checked") or {}
    sections = sorted(notebook.overview()["sections"], key=lambda s: s.get("updated") or "", reverse=True)
    floor = (date.today() - timedelta(days=MAX_BACK_DAYS)).isoformat()
    shown = set(state.get("shown") or [])
    news = [n for n in state.get("news") or [] if n["url"] not in shown]
    Database().close()   # create / migrate once before the threads
    first = (date.today() - timedelta(days=FIRST_LOOK_DAYS)).isoformat()
    asked: set[str] = set()
    for s in sections[:MAX_SECTIONS]:
        # first check of a topic: the last couple of weeks; later: since the previous check
        since = max(checked.get(s["id"]) or min(s.get("updated") or first, first), floor)
        try:
            sec = notebook.read_section(s["id"])
            query, research = topic_query(sec)
            if query.lower() in asked:   # topics on the same model share one search
                query, research = "", False
            asked.add(query.lower())
            for item in _section_news(sec, since, query, research):
                if item["url"] and item["url"] not in shown and all(n["url"] != item["url"] for n in news):
                    news.append({"section": s["id"], **item})
            checked[s["id"]] = date.today().isoformat()
        except Exception as e:   # one topic failing must not stop the others
            log.info("digest %s failed: %s", s["id"], e)
    state.update(news=news, sections_checked=checked, built=datetime.now().isoformat(timespec="minutes"),
                 shown=sorted(shown)[-2000:])
    _save(state)
    return state


def due() -> bool:
    built = _load().get("built")
    try:
        return datetime.now() - datetime.fromisoformat(built) >= timedelta(hours=EVERY_HOURS)
    except (TypeError, ValueError):
        return True


def start_background() -> None:
    """Called at server start: rebuild the digest in a thread when it is older than EVERY_HOURS."""
    if _running.is_set() or not notebook.notebook_path().exists() or not due():
        return

    def go() -> None:
        _running.set()
        try:
            build()
        except Exception as e:
            log.warning("news digest failed: %s", e)
        finally:
            _running.clear()

    threading.Thread(target=go, name="digest", daemon=True).start()


def pending() -> list[dict[str, Any]]:
    return _load().get("news") or []


def take() -> dict[str, Any] | None:
    """The unseen news, grouped by topic; marks it shown. None when there is nothing (and nothing running)."""
    state = _load()
    news = state.get("news") or []
    if not news:
        return {"status": "checking your topics in the background — call notebook_read() again in a minute"} \
            if _running.is_set() else None
    by: dict[str, list[dict[str, Any]]] = {}
    for n in news:
        by.setdefault(n["section"], []).append({k: v for k, v in n.items() if k != "section"})
    state["shown"] = sorted(set(state.get("shown") or []) | {n["url"] for n in news})[-2000:]
    state["news"] = []
    _save(state)
    return {"checked": state.get("built"), "by_topic": by,
            "how_to_use": "Tell the user briefly what is new for the topics they work on (one line each, only what "
                          "matters for their current work); fetch an item before recommending it, and add the useful "
                          "ones to the topic with notebook_add_source."}
