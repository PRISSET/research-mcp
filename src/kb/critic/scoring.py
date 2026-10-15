"""Evidence grades A-D for papers, with explicit reasons and flags."""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from kb.store import Database

TOP_VENUES = re.compile(r"\b(cvpr|iccv|eccv|neurips|nips|iclr|icml|siggraph|tpami|tog|aaai|ijcai|acl|emnlp|naacl|"
                        r"interspeech|icassp|mlsys|osdi|sosp|tmlr|wacv|bmvc|acm mm|ieee|nature|science|cell|lancet|"
                        r"nejm|jama|pnas|physical review|journal|transactions)\b", re.IGNORECASE)
CODE_RE = re.compile(r"github\.com/[\w.-]+/[\w.-]+|project page|code (is )?(available|released)|huggingface\.co/", re.IGNORECASE)
VENDOR_RE = re.compile(r"technical report|tech report|\bwe (introduce|present) (our|a family)\b", re.IGNORECASE)


def paper_signals(db: Database, paper: dict[str, Any]) -> dict[str, Any]:
    rev = db.conn.execute("SELECT status, or_venue FROM paper_reviews WHERE paper_id = ?", (paper["id"],)).fetchone()
    venue_text = " ".join(filter(None, [paper.get("venue"), paper.get("comment")]))
    text = " ".join(filter(None, [paper.get("abstract"), paper.get("comment")]))
    published = paper.get("published")
    age_months = None
    if published:
        d = date.fromisoformat(published[:10])
        age_months = (date.today().year - d.year) * 12 + date.today().month - d.month
    return {
        "openreview": dict(rev) if rev else None,
        "top_venue": bool(TOP_VENUES.search(venue_text)) or bool(rev and rev["status"] == "accepted"),
        "code": bool(CODE_RE.search(text)),
        "citations": paper.get("citation_count") or 0,
        "influential": paper.get("influential_count") or 0,
        "vendor_report": bool(VENDOR_RE.search(f"{paper.get('title', '')} {paper.get('abstract', '')}")),
        "age_months": age_months,
    }


def grade_paper(db: Database, paper: dict[str, Any]) -> dict[str, Any]:
    s = paper_signals(db, paper)
    reasons, flags = [], []
    if s["openreview"] and s["openreview"]["status"] in ("rejected_or_pending", "withdrawn"):
        flags.append(f"OpenReview: {s['openreview']['or_venue']} (not accepted there — not wrong, but not validated by review)")
    if s["vendor_report"]:
        flags.append("vendor technical report: self-evaluation, compare against independent benchmarks")
    if not s["code"]:
        flags.append("no code link found in abstract/comment")
    if s["age_months"] is not None and s["age_months"] < 6:
        flags.append(f"recent ({s['age_months']} months): citations not yet meaningful")

    reasons.append(f"{s['citations']} citations ({s['influential']} influential)")
    if s["top_venue"]:
        reasons.append("peer-reviewed venue")
    if s["code"]:
        reasons.append("code available")

    withdrawn = s["openreview"] and s["openreview"]["status"] == "withdrawn"
    if withdrawn:
        grade = "D"
    elif s["top_venue"] and s["code"] and s["influential"] >= 5:
        grade = "A"
    elif s["top_venue"] or (s["code"] and s["citations"] >= 20) or s["influential"] >= 10:
        grade = "B"
    else:
        grade = "C"
    return {"grade": grade, "reasons": reasons, "flags": flags, "signals": s}


def grade_and_save(db: Database, paper_id: str) -> dict[str, Any] | None:
    row = db.conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    if row is None:
        return None
    r = grade_paper(db, dict(row))
    _save(db, paper_id, "paper", r)
    return r


def grade_all(db: Database) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in db.conn.execute("SELECT * FROM papers").fetchall():
        r = grade_paper(db, dict(p))
        _save(db, p["id"], "paper", r)
        counts[r["grade"]] = counts.get(r["grade"], 0) + 1
    db.commit()
    return counts


def _save(db: Database, source_id: str, kind: str, r: dict[str, Any]) -> None:
    db.conn.execute(
        "INSERT OR REPLACE INTO critique (source_id, kind, grade, reasons, flags) VALUES (?, ?, ?, ?, ?)",
        (source_id, kind, r["grade"], json.dumps(r["reasons"], ensure_ascii=False), json.dumps(r["flags"], ensure_ascii=False)),
    )
