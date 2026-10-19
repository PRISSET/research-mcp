"""Web search (Brave / Tavily with a key, DuckDuckGo then Bing without) and fetch_url: read any page into the index.

fetch_url routes known sites to their APIs (arXiv -> paper full text, GitHub -> repo/issue/file, Hugging Face ->
model card, Hacker News -> whole comment tree) and extracts everything else with trafilatura (HTML) or PyMuPDF.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import threading
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from kb.critic.sources import grade_doc
from kb.research.docs import Doc, index_text, upsert_doc
from kb.research.sources import interactive_client
from kb.store import Database

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
MAX_RETURN = 12000


def web_id(url: str) -> str:
    u = urlparse(url)
    norm = f"{u.netloc.lower().removeprefix('www.')}{u.path.rstrip('/')}" + (f"?{u.query}" if u.query else "")
    return "web:" + (norm if len(norm) <= 200 else norm[:160] + "#" + hashlib.sha1(norm.encode()).hexdigest()[:10])


# ---------------- search ----------------

def _brave(query: str, limit: int) -> list[dict[str, str]]:
    http = interactive_client(0.0, {"X-Subscription-Token": os.environ["BRAVE_API_KEY"], "Accept": "application/json"})
    try:
        data = http.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": limit}).json()
    finally:
        http.close()
    return [{"title": r.get("title", ""), "url": r["url"], "snippet": re.sub(r"<[^>]+>", "", r.get("description", "")),
             "date": r.get("page_age", "")[:10] or None} for r in (data.get("web") or {}).get("results", [])]


def _tavily(query: str, limit: int) -> list[dict[str, str]]:
    http = interactive_client(0.0)
    try:
        data = http.post("https://api.tavily.com/search", json={"api_key": os.environ["TAVILY_API_KEY"], "query": query,
                                                                "max_results": limit}).json()
    finally:
        http.close()
    return [{"title": r.get("title", ""), "url": r["url"], "snippet": r.get("content", "")[:400],
             "date": (r.get("published_date") or "")[:10] or None} for r in data.get("results", [])]


def _text(node: Any) -> str:
    # Highlighted words are separate nodes: join with spaces, then squeeze.
    return " ".join(node.text(separator=" ").split())


_SCRAPE_LOCK = threading.Lock()
_last_scrape = 0.0
SCRAPE_GAP = 1.5   # seconds between keyless engine requests: bursts get "no results" pages back


def _scrape(url: str, method: str = "GET", marker: bytes = b"", **kw: Any) -> str:
    """One keyless search-engine page, spaced out process-wide; empty / bot pages are never cached."""
    global _last_scrape
    with _SCRAPE_LOCK:
        wait = SCRAPE_GAP - (time.monotonic() - _last_scrape)
        if wait > 0:
            time.sleep(wait)
        _last_scrape = time.monotonic()
    http = interactive_client(0.0, {"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.8"})
    http.cacheable = lambda resp: marker in resp.content
    try:
        return http.request(method, url, **kw).text
    finally:
        http.close()


def _duckduckgo(query: str, limit: int) -> list[dict[str, str]]:
    from selectolax.lexbor import LexborHTMLParser as HTMLParser

    html = _scrape("https://html.duckduckgo.com/html/", "POST", b"result__a", data={"q": query})
    out = []
    for res in HTMLParser(html).css("div.result"):
        a = res.css_first("a.result__a")
        if a is None:
            continue
        href = a.attributes.get("href") or ""
        if "uddg=" in href:
            href = unquote(parse_qs(urlparse(href).query).get("uddg", [""])[0])
        if not href.startswith("http") or "duckduckgo.com/y.js" in href:   # ads
            continue
        snip = res.css_first(".result__snippet")
        out.append({"title": _text(a), "url": href, "snippet": _text(snip) if snip else "",
                    "date": None})
        if len(out) >= limit:
            break
    return out


def _bing_url(href: str) -> str:
    """Bing wraps results in /ck/a?...&u=a1<base64url of the target>."""
    if "bing.com/ck/" not in href:
        return href
    u = parse_qs(urlparse(href).query).get("u", [""])[0]
    if not u.startswith("a1"):
        return ""
    try:
        return base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode()
    except (ValueError, UnicodeDecodeError):
        return ""


def _bing(query: str, limit: int) -> list[dict[str, str]]:
    """Keyless fallback when DuckDuckGo throttles us."""
    from selectolax.lexbor import LexborHTMLParser as HTMLParser

    html = _scrape("https://www.bing.com/search", marker=b"b_algo", params={"q": query, "setlang": "en"})
    out = []
    for li in HTMLParser(html).css("li.b_algo"):
        a = li.css_first("h2 a")
        href = _bing_url(a.attributes.get("href") or "") if a else ""
        if not href.startswith("http"):
            continue
        snip = li.css_first(".b_caption p") or li.css_first("p")
        out.append({"title": _text(a), "url": href, "snippet": _text(snip) if snip else "", "date": None})
        if len(out) >= limit:
            break
    return out


def _searxng(query: str, limit: int) -> list[dict[str, str]]:
    """The user's own SearXNG (metasearch over Google, Bing, Brave…): free, no limits. Needs its JSON API on."""
    http = interactive_client(0.0)
    try:
        data = http.get(os.environ["SEARXNG_URL"].rstrip("/") + "/search",
                        params={"q": query, "format": "json", "language": "en"}).json()
    finally:
        http.close()
    return [{"title": r.get("title") or r.get("url"), "url": r["url"], "snippet": r.get("content") or "",
             "date": (r.get("publishedDate") or "")[:10] or None}
            for r in data.get("results") or [] if r.get("url", "").startswith("http")][:limit]


def has_engine() -> bool:
    """A web search worth calling by default: SearXNG, Tavily or Brave (keyless scraping returns noise)."""
    return any(os.environ.get(k) for k in ("SEARXNG_URL", "TAVILY_API_KEY", "BRAVE_API_KEY"))


def web_search(db: Database, query: str, limit: int = 10) -> dict[str, Any]:
    engines = []
    if os.environ.get("SEARXNG_URL"):
        engines.append(("searxng", _searxng))
    if os.environ.get("BRAVE_API_KEY"):
        engines.append(("brave", _brave))
    if os.environ.get("TAVILY_API_KEY"):
        engines.append(("tavily", _tavily))
    engines += [("duckduckgo", _duckduckgo), ("bing", _bing)]
    errors = []
    for name, fn in engines:
        try:
            hits = fn(query, min(limit, 20))
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}")
            continue
        if not hits:
            errors.append(f"{name}: no results")
            continue
        out = []
        for h in hits:
            d = Doc(id=web_id(h["url"]), kind="web", title=h["title"], url=h["url"], platform="web",
                    summary=h["snippet"], published=h.get("date"), extra={"domain": urlparse(h["url"]).netloc})
            upsert_doc(db, d)
            g = grade_doc("web", d.extra, d.id)
            out.append({"id": d.id, "title": h["title"], "url": h["url"], "snippet": h["snippet"],
                        "date": h.get("date"), "grade": g["grade"], "flags": g["flags"]})
        db.commit()
        return {"query": query, "engine": name, "results": out,
                "next": "fetch(url=…) to read a page; if you have your own web search tool, use it too"}
    return {"query": query, "error": "web search unavailable: " + "; ".join(errors),
            "next": "use your own web search tool if you have one, then fetch the pages you find"}


# ---------------- fetch ----------------

GH_ISSUE = re.compile(r"github\.com/([^/]+/[^/]+)/(?:issues|pull|discussions)/(\d+)")
GH_BLOB = re.compile(r"github\.com/([^/]+/[^/]+)/(?:blob|tree)/[^/]+/(.+)")
GH_REPO = re.compile(r"github\.com/([^/?#]+/[^/?#]+)/?$")
HF_REPO = re.compile(r"huggingface\.co/(datasets/|spaces/)?([^/?#]+/[^/?#]+)/?$")
ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})")
HF_DISC = re.compile(r"huggingface\.co/(datasets/|spaces/)?([^/?#]+/[^/?#]+)/discussions/(\d+)")
CIVITAI_MODEL = re.compile(r"civitai\.com/models/(\d+)")
HN_ITEM = re.compile(r"news\.ycombinator\.com/item\?id=(\d+)")


def _hn_thread(db: Database, item_id: str) -> dict[str, Any]:
    http = interactive_client(0.0)
    try:
        root = http.get(f"https://hn.algolia.com/api/v1/items/{item_id}").json()
    finally:
        http.close()
    lines = [f"# {root.get('title') or 'HN thread'}", root.get("url") or "", root.get("text") or ""]

    def walk(node: dict[str, Any], depth: int) -> None:
        for c in node.get("children") or []:
            if c.get("text"):
                text = re.sub(r"<[^>]+>", " ", c["text"])
                lines.append(f"{'#' * min(depth + 2, 6)} {c.get('author')} ({(c.get('created_at') or '')[:10]})\n\n{text}")
            walk(c, depth + 1)

    walk(root, 0)
    d = Doc(id=f"hn:{item_id}", kind="hn", title=root.get("title") or f"HN {item_id}", platform="hackernews",
            url=f"https://news.ycombinator.com/item?id={item_id}", published=(root.get("created_at") or "")[:10] or None,
            extra={"points": root.get("points"), "link": root.get("url")})
    return _finish(db, d, "\n\n".join(lines), "hackernews")


def _finish(db: Database, d: Doc, text: str, source: str) -> dict[str, Any]:
    upsert_doc(db, d)
    g = grade_doc(d.kind, d.extra, d.id)
    import json

    db.conn.execute("INSERT OR REPLACE INTO critique (source_id, kind, grade, reasons, flags) VALUES (?, ?, ?, ?, ?)",
                    (d.id, d.kind, g["grade"], json.dumps(g["reasons"]), json.dumps(g["flags"])))
    idx = index_text(db, d.id, text, source, d.title)
    db.commit()
    return {"id": d.id, "kind": d.kind, "title": d.title, "url": d.url, "date": d.published, "grade": g["grade"],
            "flags": g["flags"], "indexed": idx, "text": text[:MAX_RETURN], "truncated": len(text) > MAX_RETURN,
            "outline": [m.group(0)[:100] for m in re.finditer(r"^#{1,3} .+$", text, re.M)][:60]}


def fetch_url(db: Database, url: str) -> dict[str, Any]:
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    from kb.research import platforms

    if m := ARXIV.search(url):
        from kb.process.fulltext import fetch_papers
        from kb.research.sources import arxiv_by_id

        p = arxiv_by_id(m.group(1))
        if p is None:
            return {"error": f"arXiv {m.group(1)} not found"}
        pid, _ = db.upsert_paper(p)
        db.commit()
        res = fetch_papers(db, [pid])
        text = "\n\n".join(r[0] for r in db.conn.execute(
            "SELECT '## ' || section || char(10) || text FROM chunks WHERE paper_id = ? ORDER BY ord", (pid,)))
        return {"id": pid, "kind": "paper", "title": p.title, "url": p.url, "indexed": res["papers"].get(pid),
                "text": text[:MAX_RETURN], "truncated": len(text) > MAX_RETURN}
    if m := GH_ISSUE.search(url):
        return platforms.github_read(db, m.group(1), issue=int(m.group(2)))
    if m := GH_BLOB.search(url):
        return platforms.github_read(db, m.group(1), path=m.group(2))
    if m := GH_REPO.search(url):
        return platforms.github_read(db, m.group(1))
    from kb.research.youtube import VIDEO_ID, youtube_read

    if VIDEO_ID.search(url):
        return youtube_read(db, url)
    if m := HF_DISC.search(url):
        kind = {"datasets/": "datasets", "spaces/": "spaces"}.get(m.group(1) or "", "models")
        return platforms.hf_discussion_read(db, m.group(2), int(m.group(3)), kind)
    if m := CIVITAI_MODEL.search(url):
        return platforms.civitai_read(db, m.group(1))
    if (m := HF_REPO.search(url)) and not url.rstrip("/").endswith(("/discussions", "/tree/main")):
        kind = {"datasets/": "dataset", "spaces/": "space"}.get(m.group(1) or "", "model")
        if m.group(2).split("/")[0] not in ("docs", "blog", "papers", "learn", "posts", "collections", "organizations"):
            return platforms.hf_read(db, m.group(2), kind)
    if m := HN_ITEM.search(url):
        return _hn_thread(db, m.group(1))

    if re.search(r"(^|//|\.)(reddit\.com|redd\.it)/", url):   # 403 to every non-browser client: through Redlib
        from kb.research.reddit import reddit_read

        return reddit_read(db, url)
    http = interactive_client(0.0, {"User-Agent": BROWSER_UA})
    try:
        r = http.get(url)
    finally:
        http.close()
    ctype = r.headers.get("content-type", "")
    d = Doc(id=web_id(str(r.url)), kind="web", title=url, url=str(r.url), platform="web",
            extra={"domain": urlparse(str(r.url)).netloc})
    if r.content.startswith(b"%PDF") or "pdf" in ctype:
        from kb.process.parse import parse_pdf

        blocks = parse_pdf(r.content)
        text = "\n\n".join(f"## {b.section}\n{b.text}" for b in blocks)
        d.title = blocks[0].text[:120] if blocks else url
    elif "html" in ctype or r.text.lstrip().startswith("<"):
        import trafilatura

        meta = trafilatura.extract_metadata(r.text)
        text = trafilatura.extract(r.text, output_format="markdown", include_tables=True, include_links=False,
                                   favor_recall=True) or ""
        if meta is not None:
            d.title = meta.title or url
            d.published = (meta.date or "")[:10] or None
            d.summary = meta.description or ""
            d.authors = [meta.author] if meta.author else []
            d.extra["site"] = meta.sitename
    else:
        text = r.text
    if len(text.strip()) < 200:
        return {"id": d.id, "url": d.url, "error": "page has almost no extractable text (JavaScript-only site or "
                "blocked); try your own fetch tool or another page"}
    if not d.summary:
        d.summary = " ".join(text.split())[:400]
    return _finish(db, d, text, "web")


def fetch_many(urls: list[str], per_item_chars: int = 5000) -> list[dict[str, Any]]:
    """Several pages at once (own connection per thread); texts shortened so the batch stays readable."""
    from concurrent.futures import ThreadPoolExecutor

    Database().close()   # create / migrate once before the threads

    def one(url: str) -> dict[str, Any]:
        db = Database()
        try:
            r = fetch_url(db, url)
        except Exception as e:
            return {"url": url, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        finally:
            db.close()
        for key in ("text", "card"):
            if isinstance(r.get(key), str) and len(r[key]) > per_item_chars:
                r[key] = r[key][:per_item_chars]
                r["truncated"] = True
        return r

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(urls)))) as pool:
        return list(pool.map(one, urls[:12]))
