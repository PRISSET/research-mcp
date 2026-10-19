"""Multi-platform research without network: Markdown chunking, source grades, error phrases, HF keywords,
DuckDuckGo parsing, fetch_url routing, version comparison."""

import pytest

from kb.critic.sources import grade_doc
from kb.notebook import paper_block
from kb.research import web
from kb.research.docs import Doc, parse_markdown, upsert_doc
from kb.research.platforms import _hf_keywords
from kb.research.versions import _vkey, error_core, phrase_candidates
from kb.store import Database


@pytest.fixture()
def db(tmp_path):
    database = Database(tmp_path / "kb.sqlite")
    yield database
    database.close()


README = """---
title: x
---
# Project

Intro paragraph that explains what the project does in a few words.

## Install
```bash
pip install thing==1.2
```

## Benchmarks
| GPU | it/s |
|---|---|
| A100 | 12.5 |

### Notes
It is 2.4x faster than the baseline on H100 for end-to-end inference.
"""


def test_markdown_blocks_keep_heading_path_code_and_tables():
    blocks = parse_markdown(README, "proj")
    sections = [(b.section, b.kind) for b in blocks]
    assert ("Project", "text") in sections
    assert any(b.section == "Project > Install" and "pip install thing==1.2" in b.text for b in blocks)
    table = next(b for b in blocks if b.kind == "table")
    assert table.section == "Project > Benchmarks" and "A100 | 12.5" in table.text
    assert any(b.section == "Project > Benchmarks > Notes" for b in blocks)
    assert not any("title: x" in b.text for b in blocks)                    # front matter dropped


def test_source_grades():
    assert grade_doc("repo", {"owner": "vllm-project", "stars": 50000, "pushed": "2026-10-01", "license": "Apache-2.0"})["grade"] == "A"
    stale = grade_doc("repo", {"owner": "someone", "stars": 5000, "pushed": "2022-01-01", "license": "MIT"})
    assert stale["grade"] == "C" and any("stale" in f for f in stale["flags"])
    assert grade_doc("repo", {"owner": "x", "archived": True})["grade"] == "D"
    assert grade_doc("pr", {"state": "closed", "merged": True})["grade"] == "B"
    pickled = grade_doc("model", {"owner": "rando", "downloads": 10, "pickle_files": ["model.bin"], "safetensors": False})
    assert any("pickle" in f for f in pickled["flags"]) and not any("licen" in f for f in pickled["flags"])
    assert grade_doc("model", {"owner": "minimaxai", "license": "other"})["grade"] == "A"


def test_error_core_and_phrases():
    tb = ('Traceback (most recent call last):\n  File "/x/vllm/v1/worker/gpu_model_runner.py", line 1832, in f\n'
          '    raise ValueError(...)\nValueError: Missing modality on multimodal embedding for request 0x7f3a2c')
    core = error_core(tb)
    assert core == "ValueError: Missing modality on multimodal embedding for request"
    assert phrase_candidates(core)[0] == "Missing modality on multimodal embedding"   # trailing filler dropped


def test_hf_keywords_and_versions():
    assert _hf_keywords("lip-sync talking head video generation open source")[:2] == ["lip-sync", "talking"]
    assert _hf_keywords("whisper") == []
    assert _vkey("0.10.0") > _vkey("0.9.2") and _vkey("2.14.1") > _vkey("2.4.0")


def test_duckduckgo_parsing_skips_ads(monkeypatch):
    html = """<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fy.js%3Fad">Ad</a></div>
    <div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdesign.minimax.io%2Fh3&rut=1">MiniMax <b>H3</b></a>
    <a class="result__snippet">The open <b>weights</b> are free</a></div>"""

    monkeypatch.setattr(web, "_scrape", lambda *a, **k: html)
    hits = web._duckduckgo("q", 5)
    assert hits == [{"title": "MiniMax H3", "url": "https://design.minimax.io/h3", "snippet": "The open weights are free",
                     "date": None}]


def test_bing_fallback_when_duckduckgo_is_empty(db, monkeypatch):
    import base64

    target = base64.urlsafe_b64encode(b"https://docs.vllm.ai/prefix").decode().rstrip("=")
    bing = (f'<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&p=1&u=a1{target}&ntb=1">vLLM <b>docs</b></a>'
            '</h2><div class="b_caption"><p>Prefix caching</p></div></li>')
    pages = {"duckduckgo": "<html>No results found</html>", "bing": bing}
    monkeypatch.setattr(web, "_scrape", lambda url, *a, **k: next(v for key, v in pages.items() if key in url))
    r = web.web_search(db, "vllm prefix caching", 5)
    assert r["engine"] == "bing" and r["results"][0]["url"] == "https://docs.vllm.ai/prefix"


def test_fetch_url_routes_known_sites(db, monkeypatch):
    from kb.research import platforms

    calls = []
    monkeypatch.setattr(platforms, "github_read", lambda db, repo, **kw: calls.append(("gh", repo, kw)) or {})
    monkeypatch.setattr(platforms, "hf_read", lambda db, rid, kind: calls.append(("hf", rid, kind)) or {})
    web.fetch_url(db, "https://github.com/vllm-project/vllm/issues/53700")
    web.fetch_url(db, "github.com/QwenLM/Qwen3-Omni")
    web.fetch_url(db, "https://github.com/a/b/blob/main/docs/x.md")
    web.fetch_url(db, "https://huggingface.co/MiniMaxAI/MiniMax-H3")
    web.fetch_url(db, "https://huggingface.co/datasets/org/data")
    assert calls == [("gh", "vllm-project/vllm", {"issue": 53700}), ("gh", "QwenLM/Qwen3-Omni", {}),
                     ("gh", "a/b", {"path": "docs/x.md"}), ("hf", "MiniMaxAI/MiniMax-H3", "model"),
                     ("hf", "org/data", "dataset")]


def test_docs_share_the_table_and_render_in_the_notebook(db):
    upsert_doc(db, Doc(id="gh:qwenlm/qwen3-omni", kind="repo", title="QwenLM/Qwen3-Omni", url="https://github.com/QwenLM/Qwen3-Omni",
                       platform="github", summary="Omni model", published="2025-09-01",
                       extra={"stars": 4000, "license": "Apache-2.0", "pushed": "2026-10-01"}))
    upsert_doc(db, Doc(id="gh:qwenlm/qwen3-omni", kind="repo", title="QwenLM/Qwen3-Omni", url="https://github.com/QwenLM/Qwen3-Omni",
                       platform="github", extra={"stars": 4100}))
    src = db.get_paper("gh:qwenlm/qwen3-omni")
    assert src["kind"] == "repo" and src["extra"]["stars"] == 4100 and src["extra"]["license"] == "Apache-2.0"
    assert src["abstract"] == "Omni model"                                  # not blanked by the second sighting
    block = paper_block({**src, "grade": "A"}, "- notes")
    assert "репозиторий" in block and "★ 4100" in block and "Apache-2.0" not in block and "коммит 2026-10-01" in block


def test_license_parts_are_dropped_from_cards():
    from kb.textutil import drop_licenses

    card = ("---\nlicense: other\nlicense_name: x-community\nlibrary_name: diffusers\n---\n# Model\nFast.\n"
            "## License\nTerms...\n### Commercial use\nAsk us.\n## Usage\nrun it\n")
    out = drop_licenses(card)
    assert "licen" not in out.lower() and "Commercial" not in out
    assert "library_name: diffusers" in out and "## Usage\nrun it" in out and "Fast." in out
