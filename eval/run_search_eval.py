"""Retrieval quality: recall@k / MRR of the target paper for BM25 alone and hybrid (BM25 + vectors).

Runs on the eval corpus built by eval/build_eval_db.py (the target papers plus hard distractors), so the numbers
do not depend on what the working base happens to contain. The search tool gets English queries; the Russian
questions show what happens when it does not.

Run: uv run python eval/run_search_eval.py [EVAL.sqlite]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

from kb.process.embed import embed_pending
from kb.process.search import search
from kb.store import Database

HERE = Path(__file__).parent
QUESTIONS = yaml.safe_load((HERE / "search_questions.yaml").read_text())["questions"]
K = 10


def rank_of(hits: list[dict], arxiv: str) -> int | None:
    for i, h in enumerate(hits, 1):
        if h["paper_id"] == f"arxiv:{arxiv}":
            return i
    return None


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "eval.sqlite"
    if not path.exists():
        sys.exit(f"{path} not found: build it with eval/build_eval_db.py")
    with Database(path) as db:
        db.conn.execute("UPDATE chunks SET emb = NULL")   # every run embeds with the current model
        print("vectors:", embed_pending(db))
        have = {r[0] for r in db.conn.execute("SELECT paper_id FROM paper_fulltext WHERE status = 'ok'")}
        qs = [q for q in QUESTIONS if f"arxiv:{q['arxiv']}" in have]
        print(f"scoring {len(qs)}/{len(QUESTIONS)} questions on {len(have)} papers")
        for name, dense in (("bm25", False), ("hybrid", True)):
            stats: dict[str, list[int | None]] = {"en": [], "ru": []}
            t = time.monotonic()
            for q in qs:
                stats[q["lang"]].append(rank_of(search(db, q["q"], k=K, per_paper=1, dense=dense), q["arxiv"]))
            ms = (time.monotonic() - t) * 1000 / max(1, len(qs))
            for lang, ranks in stats.items():
                if not ranks:
                    continue
                r1, r5, r10 = (sum(1 for r in ranks if r and r <= k) / len(ranks) for k in (1, 5, 10))
                mrr = sum(1 / r for r in ranks if r) / len(ranks)
                print(f"{name:8s} {lang}  n={len(ranks):3d}  R@1={r1:.2f}  R@5={r5:.2f}  R@10={r10:.2f}  "
                      f"MRR={mrr:.3f}  {ms:.0f} ms/query")


if __name__ == "__main__":
    main()
