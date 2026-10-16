"""Model choice (latest_models, model_variants, FP8 advice), release-note digests, HTTP cache, notebook refresh —
all with faked HTTP, no network."""

from datetime import date, timedelta

import httpx
import pytest

from kb.collect import cache
from kb.collect.http import RateLimitedClient
from kb.research import models, versions


class FakeHTTP:
    """Stands in for interactive_client(): answers by URL substring."""

    def __init__(self, routes):
        self.routes = routes

    def get(self, url, params=None, **kw):
        key = url + ("?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items())) if params else "")
        for needle, payload in self.routes:
            if needle in key:
                return httpx.Response(200, json=payload(params) if callable(payload) else payload,
                                      request=httpx.Request("GET", url))
        raise AssertionError(f"unexpected request {key}")

    def close(self):
        pass


def _ago(months: int) -> str:
    return (date.today() - timedelta(days=30 * months)).isoformat() + "T00:00:00.000Z"


# ---------------- precision, components, hardware ----------------

@pytest.mark.parametrize("name,label", [
    ("minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "fp8"), ("x_int8_convrot.safetensors", "int8"),
    ("Qwen3-8B-Q6_K.gguf", "q6"), ("Qwen3-8B-Q4_K_M.gguf", "4-bit"), ("model_bf16.safetensors", "bf16"),
    ("m-nvfp4.safetensors", "fp4"), ("w4a8_mixed.safetensors", "4-bit"), ("model.safetensors", "unknown"),
])
def test_precision_from_names(name, label):
    assert models.precision_of(name)[0] == label


def test_unknown_precision_is_not_treated_as_lossless():
    assert models.precision_of("model.safetensors")[1] > models.precision_of("x_fp8.safetensors")[1]


def test_components_gpus_and_fp8_support():
    assert models.component_of("vae/diffusion_pytorch_model.safetensors") == "vae"
    assert models.component_of("text_encoders/umt5_xxl_fp8.safetensors") == "text_encoder"
    assert models.component_of("diffusion_models/h3_fp8.safetensors") == "diffusion_models"
    assert models.cc_from_name("RTX 3060 Ti") == 8.6 and models.cc_from_name("H100 SXM") == 9.0
    assert models.cc_from_name("RTX 4090") == 8.9 and models.cc_from_name("RTX 5090") == 12.0
    assert "no native FP8" in models.compute_support(8.6)["fp8"] and "native FP8" in models.compute_support(8.9)["fp8"]
    assert models._fit(6.5, 8.0, 32.0).startswith("fits") and models._fit(21, 8.0, 32.0).startswith("needs")
    assert models._fit(66, 8.0, 32.0).startswith("too big")


def test_task_mapping_and_family_keys():
    assert "image-text-to-video" in models.task_tags("лучшая модель для видео")
    assert models.task_tags("speech recognition") == ["automatic-speech-recognition"]
    assert models.task_tags("text-to-image") == ["text-to-image"]
    assert models._family_key("MiniMaxAI/MiniMax-H3") == "minimax-h3"
    assert models._family_key("Wan-AI/Wan2.2-TI2V-5B") == "wan2.2"


# ---------------- latest_models ----------------

def test_latest_models_prefers_new_lab_releases(monkeypatch):
    pool = [
        {"id": "MiniMaxAI/MiniMax-H3", "createdAt": _ago(2), "likes": 6000, "tags": ["text-to-video"]},
        {"id": "someone/MiniMax-H3-hybrid", "createdAt": _ago(1), "likes": 300, "tags": []},
        {"id": "nsfw/thing", "createdAt": _ago(1), "likes": 900, "tags": ["not-for-all-audiences"]},
        {"id": "lab2/Turbo", "createdAt": _ago(1), "likes": 50, "trendingScore": 40,
         "tags": ["base_model:finetune:MiniMaxAI/MiniMax-H3"]},
        {"id": "tencent/HunyuanVideo", "createdAt": _ago(22), "likes": 2300, "tags": []},
    ]
    monkeypatch.setattr(models, "interactive_client", lambda *a, **k: FakeHTTP([("/api/models", pool)]))
    r = models.latest_models("text-to-video", months=9)
    assert [m["id"] for m in r["new_base_models"]] == ["MiniMaxAI/MiniMax-H3"]   # named-after copy is a derivative
    assert {m["id"] for m in r["hot_derivatives"]} == {"lab2/Turbo", "someone/MiniMax-H3-hybrid"}
    assert "nsfw/thing" not in str(r)
    assert r["older_but_popular"][0]["id"] == "tencent/HunyuanVideo" and r["leaderboards"]


# ---------------- model_variants ----------------

def _variants_routes():
    gb = 10**9
    repo = {"id": "org/Big", "siblings": [
        {"rfilename": "diffusion_models/big_bf16.safetensors", "size": 66 * gb},
        {"rfilename": "diffusion_models/big_fp8_scaled.safetensors", "size": 33 * gb},
        {"rfilename": "vae/vae_fp16.safetensors", "size": int(0.5 * gb)},
    ]}
    quant = [{"id": "c/Big-GGUF", "downloads": 1000}]
    gguf = {"id": "c/Big-GGUF", "siblings": [{"rfilename": "big-Q8_0.gguf", "size": 34 * gb},
                                              {"rfilename": "big-Q4_K_M.gguf", "size": 6 * gb}]}
    return [("/api/models/org/Big", repo), ("/api/models/c/Big-GGUF", gguf), ("/api/models?", quant)]


def test_variants_on_a_small_gpu_put_quality_first(monkeypatch):
    monkeypatch.setattr(models, "interactive_client", lambda *a, **k: FakeHTTP(_variants_routes()))
    monkeypatch.setattr(models, "local_hardware", lambda: {"gpus": [{"name": "RTX 3060 Ti", "vram_gb": 8.0,
                                                                    "compute_cap": 8.6}], "ram_gb": 64.0})
    r = models.model_variants("org/Big")
    assert r["advice"][0].startswith("for quality:") and "fp8" in r["advice"][0]       # offload FP8 beats Q4 in VRAM
    assert "fully in VRAM only" in r["advice"][1] and "Q4_K_M" in r["advice"][1]
    assert any("no native FP8" in a for a in r["advice"])
    assert next(v for v in r["variants"] if "vae" in v["file"])["component"] == "vae"


def test_variants_offer_fp8_when_bf16_barely_fits(monkeypatch):
    monkeypatch.setattr(models, "interactive_client", lambda *a, **k: FakeHTTP(_variants_routes()))
    monkeypatch.setattr(models, "local_hardware", lambda: {"gpus": [], "ram_gb": 64.0})
    r = models.model_variants("org/Big", vram_gb=80, gpu_name="H100")
    assert r["advice"][0].startswith("best quality that fits in VRAM") and "bf16" in r["advice"][0]
    assert "big_fp8_scaled" in r["advice"][1] and any("native FP8 compute" in a for a in r["advice"])


# ---------------- release notes ----------------

def test_changelog_finds_breaking_and_focus_lines(monkeypatch):
    rel = [{"tag_name": "v0.31.0", "body": "## Breaking changes\n- per-request mm kwargs are rejected unless a flag is set\n"
                                           "- Qwen3-Omni: fix audio in video for multimodal requests"},
           {"tag_name": "v0.30.0", "body": "- faster sampler"},
           {"tag_name": "v0.29.0", "body": "- old stuff, removed nothing"}]
    monkeypatch.setattr(versions, "interactive_client", lambda *a, **k: FakeHTTP([("/releases", rel)]))
    c = versions.changelog("vllm-project/vllm", "0.29.0", "0.31.0", ["qwen3-omni", "audio"])
    assert c["releases_between"] == ["v0.31.0", "v0.30.0"]          # yours excluded, latest included
    assert any("rejected unless" in b["line"] for b in c["breaking"])
    assert c["relevant_to_focus"][0]["line"].startswith("Qwen3-Omni: fix audio")


# ---------------- HTTP cache ----------------

def test_http_cache_serves_repeats_without_the_network(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path))
    from kb import config

    monkeypatch.setattr(cache, "settings", config.Settings())
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={"n": len(calls)})

    client = RateLimitedClient(0.0)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    a = client.get("https://api.github.com/search/repositories", params={"q": "x"}).json()
    b = client.get("https://api.github.com/search/repositories", params={"q": "x"}).json()
    c = client.get("https://api.github.com/search/repositories", params={"q": "y"}).json()
    assert a == b == {"n": 1} and c == {"n": 2} and len(calls) == 2
    assert cache.ttl_for("https://api.github.com/search/issues?q=1") == cache.HOUR
    assert cache.ttl_for("https://export.arxiv.org/api/query") == 24 * cache.HOUR
    monkeypatch.setenv("KB_HTTP_CACHE", "0")
    client.get("https://api.github.com/search/repositories", params={"q": "x"})
    assert len(calls) == 3


# ---------------- notebook refresh, batch fetch ----------------

def test_refresh_reports_only_new_things(tmp_path, monkeypatch):
    from kb import notebook
    from kb.research import find, platforms, refresh

    monkeypatch.setenv("RESEARCH_NOTEBOOK", str(tmp_path / "RESEARCH.md"))
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path / "data"))
    from kb import config

    monkeypatch.setattr("kb.store.db.settings", config.Settings())
    notebook.start("Видео", "Что нового?", ["video model"])
    since = date.today().isoformat()
    old, new = (date.today() - timedelta(days=200)).isoformat(), date.today().isoformat()

    monkeypatch.setattr(find, "find_papers", lambda db, q, n, y: {"results": []})
    monkeypatch.setattr(platforms, "hf_search", lambda db, q, k, s, n: {"results": [
        {"id": "hf:new/model", "created": new}, {"id": "hf:old/model", "created": old}]})
    monkeypatch.setattr(platforms, "github_search", lambda db, q, kind, limit: {"results": [
        {"id": f"gh:x/{kind}", "date": new}]})
    r = refresh.refresh("видео", since=since)
    assert [m["id"] for m in r["new_since"]["models"]] == ["hf:new/model"]
    assert r["new_since"]["repos"][0]["id"] == "gh:x/repos" and r["new_since"]["issues"]
    assert refresh.stale_sections() == []


def test_fetch_many_runs_in_parallel_and_trims(monkeypatch, tmp_path):
    from kb.research import web

    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path))
    from kb import config

    monkeypatch.setattr("kb.store.db.settings", config.Settings())
    monkeypatch.setattr(web, "fetch_url", lambda db, url: {"url": url, "text": "x" * 9000} if "ok" in url
                        else (_ for _ in ()).throw(RuntimeError("blocked")))
    out = web.fetch_many(["https://ok/1", "https://bad/2"], per_item_chars=100)
    assert len(out[0]["text"]) == 100 and out[0]["truncated"] and "blocked" in out[1]["error"]


# ---------------- embeddings, hybrid search, pruning ----------------

class _FakeStatic:
    """Bag-of-words vectors over a tiny vocabulary: enough to tell the hybrid search apart from BM25."""
    VOCAB = ["quantization", "int8", "attention", "sparse", "video", "cache"]

    def encode(self, texts, show_progress_bar=False):
        import numpy as np

        return np.array([[t.lower().count(w) + 0.01 for w in self.VOCAB] for t in texts], dtype=np.float32)


def _base(monkeypatch, tmp_path):
    from kb import config
    from kb.process import embed
    from kb.process.fulltext import _store
    from kb.process.parse import Block
    from kb.schemas import Paper
    from kb.store import Database

    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("kb.store.db.settings", config.Settings())
    monkeypatch.setattr(embed, "available", lambda: True)
    monkeypatch.setattr(embed, "model", lambda: _FakeStatic())
    embed._matrix.cache_clear()
    db = Database()
    for pid, text in (("web:a", "int8 quantization of attention"), ("web:b", "sparse video attention cache")):
        db.upsert_paper(Paper(id=pid, title=pid, source="web"))
        _store(db, pid, "web", [Block(section="Body", kind="text", text=text, page=None)])
    db.commit()
    return db, embed


def test_embed_now_and_hybrid_search(monkeypatch, tmp_path):
    from kb.process.search import context, search

    db, embed = _base(monkeypatch, tmp_path)
    assert embed.embed_now(db, ["web:a", "web:b"]) == "2 embedded"
    assert db.conn.execute("SELECT COUNT(*) FROM chunks WHERE emb IS NULL").fetchone()[0] == 0
    hits = search(db, "int8 quantization")
    assert hits[0]["paper_id"] == "web:a"
    assert [h["paper_id"] for h in search(db, "video cache", paper_ids={"web:b"})] == ["web:b"]
    assert context(db, hits[0]["chunk_id"])[0]["text"].startswith("int8")
    monkeypatch.setattr(embed, "MODEL", "another/model")     # a model change re-embeds everything
    assert embed.embed_now(db, ["web:a"]) == "2 embedded"
    db.close()


def test_prune_keeps_cited_and_recent(monkeypatch, tmp_path):
    from kb.store.prune import prune

    db, _ = _base(monkeypatch, tmp_path)
    db.conn.execute("CREATE TABLE h3_facts (x)")
    db.conn.execute("UPDATE paper_fulltext SET used_at = datetime('now', '-90 days')")
    db.conn.execute("UPDATE papers SET added_at = datetime('now', '-90 days')")
    db.conn.execute("INSERT INTO papers (id, title, source, added_at) VALUES "
                    "('arxiv:old', 'old', 'arxiv', datetime('now', '-90 days')), ('arxiv:new', 'new', 'arxiv', datetime('now'))")
    db.commit()
    out = prune(db, days=60, keep={"web:a"})
    assert out == {"full_texts_dropped": 1, "records_dropped": 2, "tables_dropped": ["h3_facts"], "kept_cited": 1}
    assert {r[0] for r in db.conn.execute("SELECT paper_id FROM chunks")} == {"web:a"}
    assert {r[0] for r in db.conn.execute("SELECT id FROM papers")} == {"web:a", "arxiv:new"}
    db.close()
