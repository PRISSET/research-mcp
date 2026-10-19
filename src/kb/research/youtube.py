"""YouTube: tutorials, comparisons and walkthroughs the community records instead of writing — searched without a
key and read through their transcripts.

A video is read as metadata (title, channel, views, length, publish date, description), its chapters and the
transcript in timed paragraphs ("[12:40] …"); the text is indexed, so search_fulltext finds the moment and the
citation links straight to it (…&t=760s). Manual subtitles beat auto-generated ones; English, then Russian, then
whatever exists.
"""

from __future__ import annotations

import json
import re
from typing import Any

from kb.critic.sources import grade_doc
from kb.research.docs import Doc, index_text, upsert_doc
from kb.research.sources import interactive_client
from kb.store import Database

VIDEO_ID = re.compile(r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/)|youtu\.be/)([\w-]{11})")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.8"}
PARAGRAPH_SECONDS = 45
CHAPTER = re.compile(r"^\s*\(?((?:\d{1,2}:)?\d{1,2}:\d{2})\)?\s*[-–—:|]?\s*(.+)$", re.M)


def _json_after(html: str, name: str) -> dict[str, Any]:
    m = re.search(name + r"\s*=\s*(\{.+?\});\s*(?:var |</script>)", html, re.S)
    return json.loads(m.group(1)) if m else {}


def _views(text: str | None) -> int | None:
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else None


def _stamp(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _seconds(stamp: str) -> int:
    parts = [int(p) for p in stamp.split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def _walk(o: Any, key: str) -> Any:
    if isinstance(o, dict):
        if key in o:
            yield o[key]
        for v in o.values():
            yield from _walk(v, key)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v, key)


def _text(node: dict[str, Any] | None) -> str:
    node = node or {}
    return node.get("simpleText") or "".join(r.get("text", "") for r in node.get("runs") or [])


def youtube_search(db: Database, query: str, limit: int = 6) -> dict[str, Any]:
    http = interactive_client(0.3, UA)
    try:
        html = http.get("https://www.youtube.com/results", params={"search_query": query, "hl": "en", "gl": "US"}).text
    finally:
        http.close()
    out = []
    for v in _walk(_json_after(html, "var ytInitialData"), "videoRenderer"):
        vid = v.get("videoId")
        if not vid:
            continue
        extra = {"channel": _text(v.get("ownerText")), "views": _views(_text(v.get("viewCountText"))),
                 "duration": _text(v.get("lengthText")), "age": _text(v.get("publishedTimeText")),
                 "owner": _text(v.get("ownerText"))}
        d = Doc(id=f"yt:{vid}", kind="video", title=_text(v.get("title")), url=f"https://www.youtube.com/watch?v={vid}",
                platform="youtube", summary=" ".join(_text(s.get("snippetText")) for s in v.get("detailedMetadataSnippets") or [])[:400],
                authors=[extra["channel"]], extra=extra)
        upsert_doc(db, d)
        g = grade_doc("video", extra, d.id)
        out.append({"id": d.id, "kind": "video", "title": d.title, "url": d.url, "summary": d.summary, **extra,
                    "grade": g["grade"], "flags": g["flags"]})
        if len(out) >= limit:
            break
    db.commit()
    return {"query": query, "results": out,
            "next": "fetch(url=…) reads a video: description, chapters and the transcript with timestamps"}


def _transcript(vid: str) -> tuple[list[tuple[float, str]], str]:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    found = list(api.list(vid))
    rank = {"en": 0, "ru": 1}
    found.sort(key=lambda t: (t.is_generated, rank.get(t.language_code.split("-")[0], 2)))
    t = found[0]
    snippets = t.fetch()
    return [(s.start, s.text) for s in snippets], f"{t.language_code}{' (auto)' if t.is_generated else ''}"


def youtube_read(db: Database, url: str) -> dict[str, Any]:
    m = VIDEO_ID.search(url)
    if not m:
        return {"url": url, "error": "not a YouTube video link"}
    vid = m.group(1)
    http = interactive_client(0.3, UA)
    try:
        html = http.get("https://www.youtube.com/watch", params={"v": vid, "hl": "en"}).text
    finally:
        http.close()
    player = _json_after(html, "ytInitialPlayerResponse")
    det = player.get("videoDetails") or {}
    micro = (player.get("microformat") or {}).get("playerMicroformatRenderer") or {}
    desc = det.get("shortDescription") or ""
    chapters = [(_seconds(s), title.strip()) for s, title in CHAPTER.findall(desc)]
    extra = {"channel": det.get("author"), "views": _views(det.get("viewCount")), "owner": det.get("author"),
             "duration": _stamp(int(det.get("lengthSeconds") or 0)), "published": (micro.get("publishDate") or "")[:10]}
    try:
        lines, lang = _transcript(vid)
    except Exception as e:   # no captions, or YouTube refused this IP
        lines, lang = [], f"none ({type(e).__name__})"
    parts = [f"# {det.get('title') or vid}", f"Channel: {extra['channel']} · {extra['views'] or '?'} views · "
             f"{extra['duration']} · published {extra['published'] or '?'} · transcript: {lang}", desc]
    marks = sorted(chapters)
    start, buf, ci = None, [], 0
    for sec, text in lines + [(float("inf"), "")]:
        new_chapter = ci < len(marks) and sec >= marks[ci][0]
        if buf and (new_chapter or sec - start >= PARAGRAPH_SECONDS):
            parts.append(f"[{_stamp(start)}] " + " ".join(buf))
            buf, start = [], None
        while ci < len(marks) and sec >= marks[ci][0]:
            parts.append(f"## {_stamp(marks[ci][0])} {marks[ci][1]}")
            ci += 1
        if text.strip():
            start = sec if start is None else start
            buf.append(" ".join(text.split()))
    text = "\n\n".join(p for p in parts if p)
    d = Doc(id=f"yt:{vid}", kind="video", title=det.get("title") or vid, url=f"https://www.youtube.com/watch?v={vid}",
            platform="youtube", summary=" ".join(desc.split())[:400], published=extra["published"] or None,
            authors=[extra["channel"] or ""], extra={**extra, "transcript": lang})
    upsert_doc(db, d)
    g = grade_doc("video", d.extra, d.id)
    db.conn.execute("INSERT OR REPLACE INTO critique (source_id, kind, grade, reasons, flags) VALUES (?, ?, ?, ?, ?)",
                    (d.id, "video", g["grade"], json.dumps(g["reasons"]), json.dumps(g["flags"])))
    idx = index_text(db, d.id, text, "youtube", d.title) if lines or desc else {"chunks": 0}
    db.commit()
    return {"id": d.id, "kind": "video", "title": d.title, "url": d.url, **extra, "transcript": lang,
            "grade": g["grade"], "flags": g["flags"], "indexed": idx, "chapters": [f"{_stamp(s)} {t}" for s, t in marks],
            "text": text[:12000], "truncated": len(text) > 12000,
            "cite": f"link a moment as {d.url}&t=<seconds>s (paragraphs start with [m:ss])"}
