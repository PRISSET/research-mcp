"""Build the retrieval eval corpus: the target paper of every question plus hard distractors.

Distractors are the papers a keyword search ranks highest for each question (excluding its target), plus a few
random ones, so the eval resembles a real working base of a few hundred related papers rather than the whole
collection. The chosen ids go to eval/eval_corpus.txt; the corpus is copied from an existing base (vectors left
out, so every embedding model starts equal).

Run: uv run python eval/build_eval_db.py SOURCE.sqlite [OUT.sqlite]
"""

from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
QUESTIONS = yaml.safe_load((HERE / "search_questions.yaml").read_text())["questions"]
HARD_PER_QUESTION = 6
RANDOM = 100


def main() -> None:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "data" / "eval.sqlite"
    from kb.process.search import fts_query
    from kb.store import Database

    s = sqlite3.connect(src)
    have = {r[0] for r in s.execute("SELECT paper_id FROM paper_fulltext WHERE status = 'ok'")}
    ids = {f"arxiv:{q['arxiv']}" for q in QUESTIONS} & have
    for q in QUESTIONS:
        target = f"arxiv:{q['arxiv']}"
        rows = s.execute("""SELECT c.paper_id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
                            WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, 1.0, 0.4) LIMIT 200""",
                         (fts_query(q["q"]),)).fetchall()
        hard = [r[0] for r in rows if r[0] != target and r[0] in have]
        ids.update(list(dict.fromkeys(hard))[:HARD_PER_QUESTION])
    rest = sorted(have - ids)
    ids.update(random.Random(0).sample(rest, min(RANDOM, len(rest))))
    s.close()

    (HERE / "eval_corpus.txt").write_text("\n".join(sorted(ids)) + "\n")
    out.unlink(missing_ok=True)
    db = Database(out)
    db.conn.execute("ATTACH DATABASE ? AS src", (str(src),))
    marks = ",".join("?" * len(ids))
    params = sorted(ids)
    cols = [r[1] for r in db.conn.execute("PRAGMA table_info(papers)")]
    db.conn.execute(f"INSERT INTO papers ({','.join(cols)}) SELECT {','.join(cols)} FROM src.papers WHERE id IN ({marks})", params)
    db.conn.execute(f"INSERT INTO paper_fulltext SELECT * FROM src.paper_fulltext WHERE paper_id IN ({marks})", params)
    db.conn.execute(f"""INSERT INTO chunks (id, paper_id, ord, section, kind, page, text, n_words)
                        SELECT id, paper_id, ord, section, kind, page, text, n_words FROM src.chunks
                        WHERE paper_id IN ({marks})""", params)
    db.conn.execute(f"INSERT INTO critique SELECT * FROM src.critique WHERE source_id IN ({marks})", params)
    for pid in params:
        db._reindex(pid)
    db.commit()
    n = db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    db.conn.execute("DETACH DATABASE src")
    db.close()
    print(f"{len(ids)} papers, {n} chunks -> {out}")


if __name__ == "__main__":
    main()
