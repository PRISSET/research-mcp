"""Retrieval quality: recall@k / MRR of the target paper for BM25-only, hybrid (BM25 + vectors) and hybrid + reranker.

Only questions whose target paper already has a full text are scored (the corpus is still growing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from kb.process.search import search
from kb.store import Database

QUESTIONS = yaml.safe_load((Path(__file__).with_name("search_questions.yaml")).read_text())["questions"]
K = 10


def rank_of(hits: list[dict], arxiv: str) -> int | None:
    for i, h in enumerate(hits, 1):
        if h["paper_id"] == f"arxiv:{arxiv}":
            return i
    return None


def main() -> None:
    modes = {"bm25": {"rerank": False, "dense": False}, "hybrid": {"rerank": False, "dense": True},
             "hybrid+rerank": {"rerank": True, "dense": True}}
    with Database() as db:
        have = {r[0] for r in db.conn.execute("SELECT paper_id FROM paper_fulltext WHERE status = 'ok'")}
        qs = [q for q in QUESTIONS if f"arxiv:{q['arxiv']}" in have]
        print(f"scoring {len(qs)}/{len(QUESTIONS)} questions (targets with full text)")
        for name, opt in modes.items():
            import kb.process.embed as emb

            real = emb.vector_search
            if not opt["dense"]:
                emb.vector_search = lambda *a, **k: []  # type: ignore[assignment]
            try:
                stats = {"en": [], "ru": []}
                for q in qs:
                    hits = search(db, q["q"], k=K, rerank=opt["rerank"], per_paper=1)
                    stats[q["lang"]].append(rank_of(hits, q["arxiv"]))
            finally:
                emb.vector_search = real
            for lang, ranks in stats.items():
                if not ranks:
                    continue
                r1 = sum(1 for r in ranks if r == 1) / len(ranks)
                r5 = sum(1 for r in ranks if r and r <= 5) / len(ranks)
                r10 = sum(1 for r in ranks if r and r <= 10) / len(ranks)
                mrr = sum(1 / r for r in ranks if r) / len(ranks)
                print(f"{name:14s} {lang}  n={len(ranks):3d}  R@1={r1:.2f}  R@5={r5:.2f}  R@10={r10:.2f}  MRR={mrr:.3f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
