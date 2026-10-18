"""Storage merges one work seen by several sources; source parsers; find_papers ranking without network."""

import sqlite3

import pytest

from kb.research import find as find_mod
from kb.research.sources import arxiv_query, parse_openalex, parse_s2
from kb.schemas import Paper
from kb.store import Database


@pytest.fixture()
def db(tmp_path):
    database = Database(tmp_path / "kb.sqlite")
    yield database
    database.close()


OPENALEX_WORK = {
    "id": "https://openalex.org/W123", "doi": "https://doi.org/10.48550/arxiv.2302.01318",
    "display_name": "Accelerating Large Language Model Decoding with Speculative Sampling",
    "publication_year": 2023, "publication_date": "2023-02-02", "cited_by_count": 27, "type": "preprint",
    "abstract_inverted_index": {"We": [0], "present": [1], "speculative": [2], "sampling.": [3]},
    "authorships": [{"author": {"display_name": "Charlie Chen"}}],
    "primary_location": {"landing_page_url": "https://arxiv.org/abs/2302.01318", "source": {"type": "repository",
                                                                                            "display_name": "arXiv"}},
    "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/2302.01318"},
}


def test_openalex_parsing_recovers_arxiv_and_abstract():
    p = parse_openalex(OPENALEX_WORK)
    assert p.id == "arxiv:2302.01318" and p.arxiv_id == "2302.01318" and p.openalex_id == "W123"
    assert p.abstract == "We present speculative sampling." and p.doi is None    # arXiv DOI is not a journal DOI
    assert p.venue is None and p.citation_count == 27
    journal = parse_openalex({**OPENALEX_WORK, "doi": "https://doi.org/10.1038/S41586-1", "best_oa_location": None,
                              "primary_location": {"source": {"type": "journal", "display_name": "Nature"}}})
    assert journal.id == "doi:10.1038/s41586-1" and journal.venue == "Nature" and journal.arxiv_id is None


def test_s2_parsing():
    p = parse_s2({"paperId": "abc", "title": "T", "externalIds": {"DOI": "10.1/x"}, "citationCount": 5,
                  "influentialCitationCount": 2, "tldr": {"text": "short"}, "openAccessPdf": {"url": "https://x/p.pdf"}})
    assert p.id == "doi:10.1/x" and p.s2_id == "abc" and p.tldr == "short" and p.pdf_url == "https://x/p.pdf"


def test_arxiv_query_drops_stopwords():
    assert arxiv_query("How does speculative decoding work for the LLMs?") == "all:does AND all:speculative AND all:decoding AND all:work AND all:llms"


def test_one_work_from_three_sources_is_one_row(db):
    arxiv_pid, new = db.upsert_paper(Paper(id="arxiv:2302.01318", arxiv_id="2302.01318", title="Accelerating LLM Decoding",
                                           abstract="Long arXiv abstract.", year=2023, source="arxiv"))
    assert new
    oa_pid, new = db.upsert_paper(parse_openalex({**OPENALEX_WORK, "display_name": "Accelerating LLM Decoding",
                                                  "abstract_inverted_index": None}))
    assert oa_pid == arxiv_pid and not new
    row = db.get_paper("W123")                                   # findable by OpenAlex id now
    assert row["abstract"] == "Long arXiv abstract."             # a poorer source never blanks a field
    assert row["citation_count"] == 27
    by_title, new = db.upsert_paper(Paper(id="s2:zzz", s2_id="zzz", title="accelerating llm decoding", year=2023,
                                          citation_count=900, source="s2"))
    assert by_title == arxiv_pid and not new and db.get_paper(arxiv_pid)["citation_count"] == 900
    assert db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1


def test_old_database_is_migrated(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE papers (id TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT NOT NULL DEFAULT '', "
                 "authors TEXT NOT NULL DEFAULT '[]', year INTEGER, published TEXT, updated TEXT, arxiv_id TEXT UNIQUE, "
                 "doi TEXT, s2_id TEXT, categories TEXT NOT NULL DEFAULT '[]', venue TEXT, url TEXT, pdf_url TEXT, "
                 "comment TEXT, source TEXT NOT NULL, is_seed INTEGER NOT NULL DEFAULT 0, citation_count INTEGER, "
                 "influential_count INTEGER, tldr TEXT, publication_types TEXT, s2_checked_at TEXT, "
                 "added_at TEXT NOT NULL DEFAULT (datetime('now')))")
    conn.commit()
    conn.close()
    database = Database(path)
    try:
        assert "openalex_id" in {r[1] for r in database.conn.execute("PRAGMA table_info(papers)")}
    finally:
        database.close()


def test_semantic_scholar_only_with_a_key(monkeypatch):
    monkeypatch.delenv("S2_API_KEY", raising=False)
    assert "s2" not in find_mod.default_sources()
    monkeypatch.setenv("S2_API_KEY", "k")
    assert "s2" in find_mod.default_sources()


def test_find_papers_merges_ranks_and_survives_a_dead_source(db, monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "k")
    def fake(name, query, limit, year_from):
        if name == "s2":
            raise RuntimeError("429")
        if name == "openalex":
            return [Paper(id="doi:10.1/a", doi="10.1/a", title="Shared Paper", citation_count=5000, source="openalex"),
                    Paper(id="doi:10.1/b", doi="10.1/b", title="Only OpenAlex", citation_count=1, source="openalex")]
        return [Paper(id="arxiv:9", arxiv_id="9", title="Only arXiv", source="arxiv"),
                Paper(id="doi:10.1/a", doi="10.1/a", title="Shared Paper", abstract="longer abstract", source="arxiv")]

    monkeypatch.setattr(find_mod, "_run_source", fake)
    monkeypatch.setattr(find_mod, "enrich_s2", lambda db, ids: 0)
    res = find_mod.find_papers(db, "shared paper", limit=5, on_paper=lambda pid: [])
    assert res["sources"]["s2"].startswith("failed")
    ids = [r["id"] for r in res["results"]]
    assert ids[0] == "doi:10.1/a" and len(ids) == 3                       # agreed on + well cited wins, no duplicates
    top = res["results"][0]
    assert top["found_in"] == ["openalex", "arxiv"] and top["grade"] and top["abstract"] == "longer abstract"


def test_throttled_source_is_skipped_for_a_while(db, monkeypatch):
    calls = []

    def fake(name, query, limit, year_from):
        calls.append(name)
        if name == "s2":
            raise RuntimeError("Client error '429 '")
        return []

    monkeypatch.setattr(find_mod, "_run_source", fake)
    monkeypatch.setattr(find_mod, "_COOLDOWN", {})
    find_mod.find_papers(db, "x", use=("openalex", "s2"))
    res = find_mod.find_papers(db, "y", use=("openalex", "s2"))
    assert calls.count("s2") == 1 and res["sources"]["s2"].startswith("skipped: throttled")
