"""study_topic / review_plan: checklists, facets, the readiness gate and specialist knowledge in RESEARCH.md."""

import pytest

from kb import notebook as nb
from kb.research import expertise as ex
from kb.schemas import Paper


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_NOTEBOOK", str(tmp_path / "RESEARCH.md"))
    monkeypatch.setenv("RESEARCH_LANG", "ru")
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path / "data"))
    from kb import config

    monkeypatch.setattr("kb.store.db.settings", config.Settings())


def test_profiles_and_conditional_checks():
    assert ex.profiles_for("разметка видео для LoRA") == ["annotation", "training"]
    assert ex.profiles_for("something else") == ["generic"]
    ids = {c.id for c in ex.checks_for(["annotation"], "video action annotation")}
    assert "boundaries" in ids and "captions" not in ids                   # captions only for generative data
    assert "captions" in {c.id for c in ex.checks_for(["annotation"], "caption clips for a T2V LoRA")}
    fs = ex.facets("video action annotation", ["atomic actions"], ex.checks_for(["annotation"], "video"))
    names = [f["facet"] for f in fs]
    assert names[:3] == ["foundations", "reference_works", "concept: atomic actions"] and "pitfalls" in names


def test_knowledge_is_kept_with_sources_and_gates_advice():
    from kb.store import Database

    nb.start("Разметка видео", "Как разметить атомарные движения?")
    bad = nb.add_knowledge("разметка-видео", [{"kind": "concept", "text": "Без источника нельзя писать", "sources": []}])
    assert bad["added"] == 0 and "no sources" in bad["skipped"][0]
    nb.add_knowledge("разметка-видео", [
        {"kind": "concept", "text": "Атомарное действие — минимальное движение, описываемое одним глаголом",
         "sources": ["arxiv:1705.08421"]},
        {"kind": "pitfall", "text": "Склеенные метки (действие+скорость) раздувают словарь", "sources": ["arxiv:1705.08421"]}])
    kn = nb.knowledge("разметка-видео")["knowledge"]
    assert kn["concept"][0]["sources"] == ["arxiv:1705.08421"] and len(kn["pitfall"]) == 1
    assert "arxiv:1705.08421" in nb.cited_ids()                            # never pruned from the base
    assert nb.search("атомарное")[0]["kind"] == "knowledge"
    db = Database()
    db.upsert_paper(Paper(id="arxiv:1705.08421", arxiv_id="1705.08421", title="AVA", year=2017))
    db.conn.execute("INSERT INTO chunks (paper_id, ord, section, kind, text, n_words) VALUES (?, 0, 'x', 'text', 'atomic', 1)",
                    ("arxiv:1705.08421",))
    db.commit()
    r = ex.readiness(db, "разметка-видео", "quick")
    assert r["sources_read"] == 1 and not r["expert_ready"] and any("решений" in m for m in r["missing"])
    db.close()


def test_review_plan_finds_gaps():
    from kb.store import Database

    db = Database()
    plan = "Схема разметки клипов: concept, speed, motion. Метки из закрытого списка, модель предразмечает."
    out = ex.review_plan(db, plan, topic="video annotation")
    gaps = {g["id"] for g in out["not_addressed"]}
    assert {"agreement", "vocabulary", "guidelines"} <= gaps and "prelabel" not in gaps and "warning" in out
    db.close()


def test_second_round_follows_canonical_works_and_rephrases_empty_facets(monkeypatch):
    from kb.research import expertise as ex

    monkeypatch.setattr("kb.research.web.has_engine", lambda: False)
    fs = ex.facets("video action annotation", ["atomic actions"], [])
    names = [f["facet"] for f in fs]
    per = [[] for _ in fs]
    per[names.index("reference_works")] = [
        {"kind": "paper", "title": "HMDB", "citations": 3979, "_oa": "W1"},
        {"kind": "paper", "title": "tiny", "citations": 5, "_oa": "W2"}]
    per[names.index("concept: atomic actions")] = [{"kind": "paper", "title": "AVA", "citations": 1014, "_oa": "W3"}]
    pit = names.index("pitfalls")
    fs2 = ex._second_round("video action annotation", fs, per, [pit])
    cites = [f for f in fs2 if f["papers"][0][1].startswith("cites:")]
    assert [f["papers"][0][1] for f in cites] == ["cites:W1", "cites:W3"]
    assert cites[0]["builds_on"] == "HMDB" and cites[0]["of"] == names.index("reference_works")
    again = [f for f in fs2 if f["of"] == pit]
    assert again and again[0]["papers"][0][0] != fs[pit]["papers"][0][0]
    assert "mistakes" in again[0]["papers"][0][0] and "video" in again[0]["papers"][0][0]
