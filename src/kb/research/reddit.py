"""Reddit through the user's own Redlib (`uv run kb redlib install`): search and threads with their comments.

Reddit answers 403 to scripts and closed its API to new apps; a private Redlib instance talks to it the way the
official app does and serves plain HTML, parsed here. A thread is read as the post (subreddit, author, date, score,
body) and its comments as a tree, top-sorted, each with its score — the text is indexed, so search_fulltext finds
the comment that answers the question.
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime
from typing import Any

from kb.critic.sources import grade_doc
from kb.research.docs import Doc, index_text, upsert_doc
from kb.research.sources import interactive_client
from kb.store import Database

THREAD = re.compile(r"(?:^|//|\.)(?:reddit\.com|redd\.it)(/r/\w+/comments/\w+[^?#\s]*|/\w+$)")
POST_ID = re.compile(r"/comments/(\w+)")
COMMENT = re.compile(r'<div id="(\w+)" class="comment">')
MAX_COMMENTS = 80


def base() -> str | None:
    return (os.environ.get("REDLIB_URL") or "").rstrip("/") or None


def missing() -> dict[str, Any]:
    return {"error": "Reddit blocks direct reads and has no open API",
            "next": "ask the user to run `uv run kb redlib install` in the research-mcp folder (a private Redlib on "
                    "127.0.0.1, built with cargo) — then fetch reads threads and research_sweep searches Reddit; "
                    "until then use the search snippet or your own fetch tool"}


def _get(path: str, params: dict[str, str] | None = None) -> str:
    http = interactive_client(0.3, {"Accept-Language": "en-US,en;q=0.8"})
    try:
        r = http.get(base() + path, params=params)
        if r.status_code >= 400:
            raise RuntimeError(f"Redlib answered {r.status_code} for {path}")
        return r.text
    finally:
        http.close()


def _text(fragment: str) -> str:
    """Reddit markdown rendered as HTML -> plain text with paragraphs, list items and link targets."""
    s = re.sub(r'<a [^>]*href="([^"]+)"[^>]*>(.*?)</a>',
               lambda m: m.group(2) if m.group(1) in (m.group(2), html.unescape(m.group(2)))
               else f"{m.group(2)} ({m.group(1)})", fragment, flags=re.S)
    s = re.sub(r"<li>", "- ", s)
    s = re.sub(r"</(p|li|h\d|pre|blockquote|tr)>|<br\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", s)).strip()


def _date(title: str | None) -> str | None:
    try:
        return datetime.strptime(title or "", "%b %d %Y, %H:%M:%S UTC").date().isoformat()
    except ValueError:
        return None


def _first(pattern: str, s: str) -> str | None:
    m = re.search(pattern, s, re.S)
    return m.group(1) if m else None


def _num(s: str | None) -> int | None:
    try:
        return int(s) if s is not None else None
    except ValueError:
        return None


def parse_posts(page: str) -> list[dict[str, Any]]:
    out = []
    for block in re.split(r'<div class="post(?: [\w ]*)?" id="', page)[1:]:
        link = _first(r'<h2 class="post_title">.*?<a href="(/r/\w+/comments/[^"]+)">(?:.*?)</a>\s*</h2>', block)
        title = _first(r'<h2 class="post_title">.*?<a href="/r/\w+/comments/[^"]+">(.*?)</a>\s*</h2>', block)
        if not link or not title:
            continue
        out.append({
            "post_id": block.split('"', 1)[0], "path": link, "title": html.unescape(title).strip(),
            "subreddit": _first(r'class="post_subreddit" href="/(r/\w+)"', block),
            "author": _first(r'class="post_author[^"]*" href="/u(?:ser)?/([\w-]+)"', block),
            "date": _date(_first(r'class="created" title="([^"]+)"', block)),
            "score": _num(_first(r'class="post_score" title="(-?\d+)"', block)),
            "comments_n": _num(_first(r'title="(\d+) comments?"', block)),
            "flair": html.unescape(_first(r'class="post_flair"[^>]*><span>(.*?)</span>', block) or "") or None,
            "preview": _text(_first(r'<div class="post_body post_preview">(.*?)</div>\s*</div>', block) or "")[:400],
        })
    return out


def _doc(p: dict[str, Any], summary: str, **extra: Any) -> Doc:
    pid = p.get("post_id") or POST_ID.search(p["path"]).group(1)
    return Doc(id=f"reddit:{pid}", kind="reddit", title=f"{p.get('subreddit')}: {p['title']}",
               url="https://www.reddit.com" + p["path"], platform="reddit", summary=summary[:400],
               published=p.get("date"), authors=[p.get("author") or ""],
               extra={"subreddit": p.get("subreddit"), "score": p.get("score"), "comments_n": p.get("comments_n"),
                      "flair": p.get("flair"), **extra})


def reddit_search(db: Database, query: str, limit: int = 6, subreddit: str | None = None,
                  period: str = "all") -> dict[str, Any]:
    """Reddit posts for a query, most relevant first; `subreddit` limits it to one (e.g. "StableDiffusion")."""
    if not base():
        return {"query": query, "results": [], **missing()}
    path = f"/r/{subreddit.removeprefix('r/')}/search" if subreddit else "/search"
    params = {"q": query, "sort": "relevance", "t": period, **({"restrict_sr": "on"} if subreddit else {})}
    out = []
    for p in parse_posts(_get(path, params))[:limit]:
        d = _doc(p, p["preview"])
        upsert_doc(db, d)
        g = grade_doc("reddit", d.extra, d.id)
        out.append({"id": d.id, "kind": "reddit", "title": d.title, "url": d.url, "date": d.published,
                    "score": p["score"], "comments_n": p["comments_n"], "flair": p["flair"], "summary": p["preview"],
                    "grade": g["grade"], "flags": g["flags"]})
    db.commit()
    return {"query": query, "results": out,
            "next": "fetch(url=…) reads a thread with its comments (top first, with scores)"}


def reddit_read(db: Database, url: str) -> dict[str, Any]:
    if not base():
        return {"url": url, **missing()}
    m = THREAD.search(url)
    if not m:
        return {"url": url, "error": "not a Reddit thread link (…/r/<sub>/comments/<id>/…)"}
    page = _get(m.group(1), {"sort": "top"})
    head = page[page.find('<div class="post highlighted'):]
    flair = _first(r'class="post_flair"[^>]*><span>(.*?)</span>', head)
    title = _text(re.sub(r'<a [^>]*class="post_flair".*?</a>', "", _first(r'<h1 class="post_title">(.*?)</h1>', head)
                         or "", flags=re.S))
    path = (_first(r'id="reddit_url">https://www\.reddit\.com(/[^<]+)</p>', page) or m.group(1)).split("?")[0]
    p = {"path": path, "title": title or "Reddit thread", "flair": html.unescape(flair) if flair else None,
         "subreddit": _first(r'class="post_subreddit" href="/(r/\w+)"', head),
         "author": _first(r'class="post_author[^"]*" href="/u(?:ser)?/([\w-]+)"', head),
         "date": _date(_first(r'class="created" title="([^"]+)"', head)),
         "score": _num(_first(r'class="post_score" title="(-?\d+)"', head))}
    body = _text(_first(r'<div class="post_body">(.*?)<div class="post_score"', head) or "")
    comments = []
    marks = list(COMMENT.finditer(page))
    for i, c in enumerate(marks[:MAX_COMMENTS]):
        seg = page[c.start():marks[i + 1].start() if i + 1 < len(marks) else len(page)]
        before = page[:c.start()]
        depth = before.count('<blockquote class="replies">') - before.count("</blockquote>")
        text = _text(_first(r'<div class="comment_body[^"]*">(.*?)</div></div>', seg) or "")
        if not text:
            continue
        comments.append({"depth": max(depth, 0), "author": _first(r'class="comment_author[^"]*" href="/u(?:ser)?/([\w-]+)"', seg),
                         "score": _num(_first(r'class="comment_score" title="(-?\d+)"', seg)),
                         "date": _date(_first(r'class="created" title="([^"]+)"', seg)), "text": text,
                         "op": p["author"] is not None and f'href="/user/{p["author"]}"' in seg.split("</summary>")[0]})
    p["comments_n"] = len(comments)
    parts = [f"# {p['title']}", f"{p['subreddit']} · u/{p['author']} · {p['date']} · {p['score']} upvotes"
             + (f" · {p['flair']}" if p["flair"] else ""), body]
    for c in comments:
        tag = " (OP)" if c["op"] else ""
        parts.append(f"{'  ' * c['depth']}[{c['score']}] u/{c['author']}{tag}: {c['text']}")
    text = "\n\n".join(x for x in parts if x)
    d = _doc(p, body or (comments[0]["text"] if comments else ""), top_comment_score=max(
        (c["score"] or 0 for c in comments), default=0))
    upsert_doc(db, d)
    g = grade_doc("reddit", d.extra, d.id)
    db.conn.execute("INSERT OR REPLACE INTO critique (source_id, kind, grade, reasons, flags) VALUES (?, ?, ?, ?, ?)",
                    (d.id, "reddit", g["grade"], json.dumps(g["reasons"]), json.dumps(g["flags"])))
    idx = index_text(db, d.id, text, "reddit", d.title)
    db.commit()
    return {"id": d.id, "kind": "reddit", "title": d.title, "url": d.url, "date": p["date"], "score": p["score"],
            "comments_n": len(comments), "grade": g["grade"], "flags": g["flags"], "indexed": idx,
            "text": text[:12000], "truncated": len(text) > 12000}
