"""find_methods ranking / claim extraction and research_brief questions — no network, no models."""

from datetime import date

from kb.research import brief, methods
from kb.schemas import Paper


def test_expand_swaps_synonyms():
    qs = methods.expand("x", ["training-free ODE solver flow matching"])
    assert "training-free ODE sampler flow matching" in qs and "training-free ODE solver rectified flow" in qs
    assert methods.expand("улучшить солвер", None) == []          # non-English goal needs explicit queries


def test_claims_prefer_the_methods_own_result():
    abstract = ("However, existing solvers degrade badly at 10 NFE. We propose A. "
                "With only 5 NFE, we achieve 6.61 FID on CIFAR-10. It is simple.")
    assert methods.claims(abstract) == ["With only 5 NFE, we achieve 6.61 FID on CIFAR-10."]


def test_profile_and_score_reward_evidence():
    strong = Paper(id="arxiv:1", title="A-FloPS: adaptive flow path sampler", venue="AAAI",
                   abstract="A training-free sampler. We reduce FID by 30% at 5 NFE with no additional cost. "
                            "Code: https://github.com/x/afloPS.")
    survey = Paper(id="arxiv:2", title="A survey of diffusion samplers", abstract="We review samplers.")
    ps, pv = methods.profile(strong), methods.profile(survey)
    assert ps["training_free"] and ps["top_venue"] and ps["code"] == "https://github.com/x/afloPS" and ps["claims"]
    assert methods.score(0.7, ps, strong, date.today()) > methods.score(0.7, pv, survey, date.today())


def test_find_methods_merges_versions_and_ranks(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path))
    from kb import config

    monkeypatch.setattr("kb.store.db.settings", config.Settings())
    from kb.store import Database

    a1 = Paper(id="arxiv:2509.00036", arxiv_id="2509.00036", title="A-FloPS: Adaptive Flow Path Sampler",
               abstract="Training-free. We achieve 20% lower FID at 5 NFE.", source="openalex")
    a2 = Paper(id="doi:10.1609/x", doi="10.1609/x", title="A-FloPS: adaptive flow path sampler", venue="AAAI",
               abstract="Training-free.", citation_count=7, source="openalex")
    off = Paper(id="arxiv:9", arxiv_id="9", title="Histone H3 variants", abstract="Biology.", source="arxiv")
    monkeypatch.setattr(methods, "_openalex", lambda q, y: [a1, a2])
    monkeypatch.setattr(methods.sources, "arxiv_search", lambda q, n, y: [off])
    monkeypatch.setattr(methods, "_baseline_citers", lambda b, y: [])
    monkeypatch.setattr(methods, "_semantic", lambda g, ps: [0.9 if "FloPS" in p.title else 0.1 for p in ps])
    r = methods.find_methods(Database(), "better sampler", ["flow matching sampler"], ["UniPC"])
    assert [x["title"] for x in r["results"]].count("A-FloPS: Adaptive Flow Path Sampler") == 1
    top = r["results"][0]
    assert top["id"] == "arxiv:2509.00036" and top["venue"] == "AAAI" and top["claimed_gains"]


def test_brief_asks_until_ready(monkeypatch):
    monkeypatch.setenv("RESEARCH_LANG", "ru")
    r = brief.research_brief("Найди солвер лучше res_multistep для MiniMax H3 без потери скорости")
    assert not r["ready"] and {"goal", "environment", "baseline", "success_metric", "constraints"} <= set(r["missing"])
    assert any("видеокарт" in q["question"] for q in r["questions"])
    r = brief.research_brief(
        "Найди солвер лучше res_multistep для MiniMax H3 без потери скорости",
        goal="лучшее качество видео и звука при тех же 20 шагах", environment="ComfyUI, MiniMax H3, RTX 4090 24 GB",
        baseline="res_multistep, simple, 20 шагов", success_metric="детализация видео и разборчивость речи, без артефактов",
        constraints="замедление не больше 5 %, без дообучения")
    assert r["ready"] and "res_multistep" in r["brief"]
    assert not brief.research_brief("x", goal="не знаю")["ready"]
