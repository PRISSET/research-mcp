"""estimate_resources / compare_models / dataset_info / paper_code with faked HTTP."""

import httpx
import pytest

from kb.research import ml

QWEN_CFG = {"architectures": ["Qwen3ForCausalLM"], "hidden_size": 4096, "num_hidden_layers": 36,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12288,
            "vocab_size": 151936, "max_position_embeddings": 40960}
MOE_CFG = {**QWEN_CFG, "hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 32,
           "num_key_value_heads": 4, "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768}


class Fake:
    def __init__(self, routes):
        self.routes = routes

    def get(self, url, params=None, **kw):
        for needle, payload in self.routes:
            if needle in url:
                if isinstance(payload, Exception):
                    raise payload
                if isinstance(payload, str):
                    return httpx.Response(200, text=payload, request=httpx.Request("GET", url))
                return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
        raise httpx.HTTPError(f"unexpected {url}")

    def close(self):
        pass


@pytest.fixture
def hf(monkeypatch):
    def install(routes):
        monkeypatch.setattr(ml, "interactive_client", lambda *a, **k: Fake(routes))
        monkeypatch.setattr(ml, "local_hardware", lambda: {"gpus": [], "ram_gb": 32})
    return install


def test_inference_kv_cache_and_speed(hf):
    hf([("/api/models/Qwen/Qwen3-8B", {"id": "Qwen/Qwen3-8B", "safetensors": {"total": 8_190_735_360}}),
        ("/Qwen/Qwen3-8B/resolve/main/config.json", QWEN_CFG)])
    r = ml.estimate_resources("Qwen/Qwen3-8B", seq_len=32768, gpu="RTX 4090", vram_gb=24)
    assert r["breakdown_gb"]["kv_cache"] == pytest.approx(4.5, abs=0.05)     # 36 x 8 x 128 x 2 x 2 B per token
    assert r["breakdown_gb"]["weights"] == pytest.approx(15.26, abs=0.05) and r["fits"] is True
    assert 35 < r["decode_tokens_per_s_batch1"] < 40        # 1008 GB/s x 0.7 / (16.4 GB weights + 2.4 GB avg KV)
    fp8kv = ml.estimate_resources("Qwen/Qwen3-8B", seq_len=32768, kv_cache="fp8", gpu="RTX 4090", vram_gb=24)
    assert fp8kv["breakdown_gb"]["kv_cache"] == pytest.approx(2.25, abs=0.05)


def test_training_modes(hf):
    hf([("/api/models/Qwen/Qwen3-8B", {"id": "Qwen/Qwen3-8B", "safetensors": {"total": 8_190_735_360}}),
        ("/Qwen/Qwen3-8B/resolve/main/config.json", QWEN_CFG)])
    q = ml.estimate_resources("Qwen/Qwen3-8B", mode="qlora", seq_len=2048, gpu="RTX 3060 Ti", vram_gb=8,
                              train_tokens=5e7)
    full = ml.estimate_resources("Qwen/Qwen3-8B", mode="full", gpu="H100", vram_gb=80)
    sharded = ml.estimate_resources("Qwen/Qwen3-8B", mode="full", gpu="H100", vram_gb=80, num_gpus=8)
    assert q["precision"] == "nf4" and q["breakdown_gb"]["weights"] < 4.5 and q["train_time_hours"] > 0
    assert full["breakdown_gb"]["adam_states_fp32"] == pytest.approx(91.5, abs=0.5) and not full["fits"]
    assert sharded["fits"] and any("FSDP" in a for a in sharded["assumptions"])


def test_moe_active_params(hf):
    hf([("/api/models/Qwen/Qwen3-30B-A3B", {"id": "Qwen/Qwen3-30B-A3B", "safetensors": {"total": 30_532_122_624}}),
        ("/Qwen/Qwen3-30B-A3B/resolve/main/config.json", MOE_CFG)])
    f = ml.model_facts("Qwen/Qwen3-30B-A3B")
    assert 3.0e9 < f["active_params"] < 3.7e9


def test_gated_config_comes_from_the_mirror(hf):
    hf([("/api/models/meta-llama/Llama-3.1-8B", {"id": "meta-llama/Llama-3.1-8B", "gated": "manual"}),
        ("/meta-llama/Llama-3.1-8B/resolve/main/config.json", httpx.HTTPError("401")),
        ("/unsloth/Llama-3.1-8B/resolve/main/config.json", {**QWEN_CFG, "max_position_embeddings": 131072})])
    f = ml.model_facts("meta-llama/Llama-3.1-8B")
    assert f["context"] == 131072 and f["config_from"] == "unsloth/Llama-3.1-8B"


def test_card_tables_markdown_and_html_rowspan():
    md = "| Benchmark | A | B |\n|---|---|---|\n| MMLU | 70.1 | 68.0 |\n| Speed | fast | slow |\n"
    html = ("<table><tr><td>Category</td><td>Benchmark</td><td>8B</td></tr>\n<tr><td rowspan=2>General\n</td>"
            "<td>MMLU\n</td><td>66.7</td></tr>\n<tr><td>GPQA</td><td>30.4</td></tr></table>")
    rows = ml.benchmark_rows(md + "\n" + html)
    assert rows[0]["rows"] == [["MMLU", "70.1", "68.0"]]
    assert rows[1]["rows"] == [["General", "MMLU", "66.7"], ["", "GPQA", "30.4"]]


def test_dataset_info(hf):
    hf([("/api/datasets/openai/gsm8k", {"id": "openai/gsm8k", "cardData": {"license": "mit"}, "tags": ["language:en"]}),
        ("datasets-server.huggingface.co/size", {"size": {"dataset": {"num_rows": 17584, "num_bytes_parquet_files": 5e6},
                                                          "configs": [{"config": "main", "num_rows": 8792}],
                                                          "splits": [{"config": "main", "split": "train", "num_rows": 7473}]}}),
        ("datasets-server.huggingface.co/first-rows", {"features": [{"name": "question", "type": {"dtype": "string"}}],
                                                       "rows": [{"row": {"question": "Natalia sold clips"}}]}),
        ("/api/models", [{"id": "x/math-model", "downloads": 10}])])
    r = ml.dataset_info("https://huggingface.co/datasets/openai/gsm8k")
    assert r["total_rows"] == 17584 and r["columns"] == {"question": "string"}
    assert r["sample"][0]["question"].startswith("Natalia") and r["models_trained_on_it"][0]["id"] == "x/math-model"


def test_paper_code_finds_the_official_repo(hf):
    hf([("/api/papers/2509.00036", {"error": "Paper not found"}), ("/api/arxiv/2509.00036/repos", {}),
        ("api.github.com/search/repositories", {"items": [
            {"full_name": "someone/awesome-diffusion", "stargazers_count": 900, "html_url": "u1", "description": "list"},
            {"full_name": "other/notes", "stargazers_count": 50, "html_url": "u2", "description": ""},
            {"full_name": "jinc7461/AFloPS", "stargazers_count": 11, "html_url": "u3", "description": "Official code"}]})])
    import kb.research.sources as src

    from kb.schemas import Paper

    orig = src.arxiv_by_id
    src.arxiv_by_id = lambda aid: Paper(id="arxiv:2509.00036", title="A-FloPS: Accelerating Diffusion Models")
    try:
        r = ml.paper_code("https://arxiv.org/abs/2509.00036v2")
    finally:
        src.arxiv_by_id = orig
    assert r["github_mentions"][0]["repo"] == "jinc7461/AFloPS" and r["github_mentions"][0]["likely_official"]
    assert all("awesome" not in m["repo"] for m in r["github_mentions"])
