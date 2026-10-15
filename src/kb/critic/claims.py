"""verify_claim: gather what supports or disputes a statement from the local paper base, graded by evidence."""

from __future__ import annotations

import re
from typing import Any

from kb.store import Database


def _terms(text: str) -> list[str]:
    stop = {"the", "and", "with", "that", "this", "than", "from", "faster", "better", "is", "are", "for", "на", "для", "что", "это"}
    return [w for w in re.findall(r"[\w.+-]{3,}", text.lower()) if w not in stop]


def verify_claim(db: Database, claim: str, paper_ids: set[str] | None = None) -> dict[str, Any]:
    terms = _terms(claim)
    papers = db.search(" ".join(terms[:8]), paper_ids=paper_ids, limit=6) if terms else []
    graded = []
    for p in papers:
        row = db.conn.execute("SELECT grade FROM critique WHERE source_id = ?", (p["id"],)).fetchone()
        graded.append({"paper": p["title"], "id": p["id"], "year": p["year"], "grade": row["grade"] if row else None,
                       "tldr": p.get("tldr")})

    passages = []
    try:  # evidence from full texts; empty if nothing was fetched yet
        from kb.process.search import search as fulltext_search

        passages = [{k: h[k] for k in ("citation", "grade", "text", "url", "chunk_id", "score")}
                    for h in fulltext_search(db, claim, k=6, paper_ids=paper_ids)]
    except Exception:
        pass
    numbers = []
    try:
        from kb.process.claims import find as find_numbers

        numbers = [{k: n[k] for k in ("kind", "value", "unit", "hardware", "models", "sentence", "title", "grade")}
                   for n in find_numbers(db, text=" ".join(terms[:3]), limit=8)]
    except Exception:
        pass

    strong = [g for g in graded if g["grade"] in ("A", "B")]
    if not graded and not passages:
        verdict = "no evidence in the local base — run find_papers and fetch_fulltext on the topic first"
    elif strong:
        verdict = "related A/B-graded papers found — read the passages to see whether they support or contradict it"
    else:
        verdict = "only weak (C/D) or ungraded sources — treat the claim as unverified"
    return {
        "claim": claim, "verdict": verdict, "related_papers": graded,
        "fulltext_passages": passages, "measured_numbers": numbers,
        "note": "Retrieval, not entailment: read fulltext_passages with their conditions (hardware, model, dataset, "
                "settings) before concluding. A number measured under other conditions does not transfer.",
    }
