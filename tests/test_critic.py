"""Critic: evidence grades, OpenReview classes, citation stance, verify_claim."""

import pytest

from kb.critic.citation_contexts import method_name, stance
from kb.critic.claims import verify_claim
from kb.critic.openreview import classify_venue
from kb.critic.scoring import grade_and_save, grade_paper
from kb.schemas import Paper
from kb.store import Database


@pytest.fixture()
def db(tmp_path):
    database = Database(tmp_path / "kb.sqlite")
    yield database
    database.close()


def test_openreview_venue_classes():
    assert classify_venue("ICLR 2025 Poster") == "accepted"
    assert classify_venue("Submitted to ICLR 2025") == "rejected_or_pending"
    assert classify_venue("ICLR 2025 Conference Withdrawn Submission") == "withdrawn"


def test_paper_grades(db):
    db.upsert_paper(Paper(id="arxiv:1", title="Strong", abstract="Code: github.com/a/b", venue="CVPR 2024", published="2023-01-01"))
    db.conn.execute("UPDATE papers SET citation_count = 300, influential_count = 40 WHERE id = 'arxiv:1'")
    db.upsert_paper(Paper(id="arxiv:2", title="Fresh technical report", abstract="We present our model.", published="2026-09-01"))
    strong, fresh = (grade_paper(db, db.get_paper(i)) for i in ("arxiv:1", "arxiv:2"))
    assert strong["grade"] == "A"
    assert fresh["grade"] == "C" and any("vendor" in f for f in fresh["flags"]) and any("recent" in f for f in fresh["flags"])


def test_journal_papers_count_as_peer_reviewed(db):
    db.upsert_paper(Paper(id="doi:10.1038/x", title="Protein folding", doi="10.1038/x", venue="Nature",
                          published="2021-07-15", citation_count=5000, source="openalex"))
    assert grade_and_save(db, "doi:10.1038/x")["grade"] in ("A", "B")


def test_citation_stance_is_directional():
    assert method_name("SageAttention: Accurate 8-Bit Attention") == "SageAttention"
    assert stance("SageAttention [5] fails to preserve accuracy at 4 bits.", "SageAttention") == "critical"
    assert stance("We apply a short LoRA [7] step to recover part of the quality loss.", "LoRA") == "possible_critical"
    assert stance("Our method outperforms TeaCache by 1.3x.", "TeaCache") == "comparative"


def test_verify_claim_on_an_empty_base_says_so(db):
    v = verify_claim(db, "4-bit quantization loses less than 1% on MMLU")
    assert v["verdict"].startswith("no evidence") and v["related_papers"] == []
