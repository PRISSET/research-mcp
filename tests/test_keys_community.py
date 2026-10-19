"""Optional keys file, the experiments journal, Civitai / ComfyUI / HF discussion parsing — no network."""

import os

import pytest

from kb import keys
from kb import notebook as nb


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_KEYS", str(tmp_path / "keys.env"))
    monkeypatch.setenv("RESEARCH_NOTEBOOK", str(tmp_path / "RESEARCH.md"))
    monkeypatch.setenv("RESEARCH_LANG", "ru")
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path / "data"))
    from kb import config

    monkeypatch.setattr("kb.store.db.settings", config.Settings())
    for k in keys.NAMES:
        monkeypatch.delenv(k, raising=False)
    keys._HINTED.clear()


def test_keys_file_is_private_and_loaded_without_overriding(tmp_path, monkeypatch):
    p = keys.set_key("TAVILY_API_KEY", "tvly-1")
    keys.set_key("GITHUB_TOKEN", "gh-1")
    keys.set_key("GITHUB_TOKEN", "gh-2")                                  # replaced, not duplicated
    assert oct(p.stat().st_mode & 0o777) == "0o600" and p.read_text().count("GITHUB_TOKEN") == 1
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    keys.load()
    assert os.environ["TAVILY_API_KEY"] == "from-env" and os.environ["GITHUB_TOKEN"] == "gh-2"
    monkeypatch.delenv("GITHUB_TOKEN")
    hint = keys.missing_hint(["GITHUB_TOKEN"])
    assert "github.com/settings/tokens" in hint and keys.missing_hint(["GITHUB_TOKEN"]) is None   # once


def test_experiments_are_logged_and_read_back():
    nb.start("Шот H3", "как ускорить")
    nb.add_experiment("шот-h3", "SageAttention 2 | вместо sdpa", "6.8 → 4.9 мин", "better", "10 шагов")
    r = nb.add_experiment("шот-h3", "TeaCache 0.2", "артефакты на руках", "worse")
    assert r["experiments"] == 2 and r["failed_or_worse"] == ["TeaCache 0.2"]
    exps = nb.read_section("шот-h3")["experiments"]
    assert exps[0]["change"] == "SageAttention 2 | вместо sdpa" and exps[0]["verdict"] == "better"
    assert nb.search("teacache")[0]["kind"] == "experiment"
    with pytest.raises(nb.NotebookError):
        nb.add_experiment("шот-h3", "x", "y", "great")


def test_civitai_node_and_discussion_docs_are_graded():
    from kb.critic.sources import grade_doc
    from kb.research.platforms import _civitai_doc, _fallbacks

    d = _civitai_doc({"id": 7, "name": "H3 speed workflow", "type": "Workflows", "nsfw": False,
                      "description": "<p>Fast <b>H3</b></p>", "stats": {"downloadCount": 20000, "thumbsUpCount": 900},
                      "creator": {"username": "u"},
                      "modelVersions": [{"name": "v2", "baseModel": "MiniMax H3", "publishedAt": "2026-09-01T00:00:00Z",
                                         "trainedWords": ["m1la"], "files": [{"name": "w.safetensors"}]}]})
    assert d.id == "civitai:7" and d.summary == "Fast H3" and d.extra["base_models"] == ["MiniMax H3"]
    assert grade_doc("civitai", d.extra)["grade"] == "B"
    assert grade_doc("node", {"stars": 10, "status": "NodeVersionStatusFlagged"})["grade"] == "D"
    assert "not confirmed" in grade_doc("discussion", {"repo": "a/b"})["flags"][0]
    assert _fallbacks("qwen image lora training")[0] == "qwen image lora training"


def test_sweep_without_a_web_key_hands_queries_to_the_agent(monkeypatch):
    from kb.research import sweep

    monkeypatch.setattr("kb.research.web.web_search", lambda db, q, n: {"results": [{"title": "scraped"}]})
    assert sweep.research_sweep("qwen image lora", platforms=["web"])["web"] == [{"title": "scraped"}]   # forced
    monkeypatch.setattr(sweep, "community", lambda q, parts, n: {p: [] for p in parts})
    out = sweep.research_sweep("qwen image lora", platforms=["civitai"])
    assert set(out) >= {"community"} and "web" not in out
    assert "site:reddit.com" in sweep.web_queries("x")["use_your_own_search"][1]


def test_hf_token_goes_only_to_hugging_face(monkeypatch):
    import httpx

    from kb.collect.http import RateLimitedClient

    seen = {}
    c = RateLimitedClient(0.0, use_cache=False)
    c._client = httpx.Client(transport=httpx.MockTransport(
        lambda r: seen.__setitem__(r.url.host, r.headers.get("authorization")) or httpx.Response(200)))
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    c.get("https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/config.json")
    c.get("https://api.github.com/repos/a/b")
    assert seen == {"huggingface.co": "Bearer hf_x", "api.github.com": None}
