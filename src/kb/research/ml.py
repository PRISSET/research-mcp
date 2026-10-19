"""Everyday ML questions answered from live data, not from memory:

- estimate_resources: VRAM / RAM for inference, LoRA, QLoRA or full fine-tuning of a Hugging Face model (from its
  config.json and real parameter count), training time for N tokens and decode speed on a given GPU;
- compare_models: architecture, size, context, dates and the benchmark rows from each model card side by side;
- dataset_info: splits, rows, size, columns, sample rows, and the models trained on it;
- paper_code: the official code, project page and Hugging Face models / datasets / Spaces for a paper.

All estimates are formulas with stated assumptions — the output says which, so the agent can tell the user.
"""

from __future__ import annotations

import html as htmllib
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from kb.research.models import HF, cc_from_name, local_hardware
from kb.research.sources import interactive_client

GB = 1024**3

# ---------------- model facts ----------------

LLM_KEYS = ("text_config", "llm_config", "language_config", "thinker_config")


def _repo(model_id: str) -> str:
    return re.sub(r"^(hf:|https?://huggingface\.co/)", "", model_id.strip()).strip("/")


def _text_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Multimodal configs nest the language model; its numbers drive memory."""
    for k in LLM_KEYS:
        sub = cfg.get(k)
        if isinstance(sub, dict):
            inner = _text_cfg(sub)
            if inner.get("hidden_size"):
                return inner
    return cfg


def model_facts(model_id: str, http: Any = None) -> dict[str, Any]:
    rid = _repo(model_id)
    own = http is None
    http = http or interactive_client(0.05)
    try:
        meta = http.get(f"{HF}/api/models/{rid}").json()
        if "error" in meta:
            return {"id": rid, "error": meta["error"]}
        cfg, cfg_from = {}, rid
        for src in (rid, f"unsloth/{rid.split('/')[-1]}") if meta.get("gated") else (rid,):
            try:   # gated repos: the unsloth mirror carries the same config.json without a login
                cfg = http.get(f"{HF}/{src}/resolve/main/config.json").json()
                cfg_from = src
                break
            except Exception:
                continue
        if not cfg:   # a diffusers repo without a top-level config: the partial one from the API
            cfg = meta.get("config") or {}
    finally:
        if own:
            http.close()
    t = _text_cfg(cfg)
    st = meta.get("safetensors") or {}
    h, layers = t.get("hidden_size") or t.get("d_model"), t.get("num_hidden_layers") or t.get("n_layer")
    heads = t.get("num_attention_heads") or t.get("n_head")
    kv_heads = t.get("num_key_value_heads") or heads
    head_dim = t.get("head_dim") or (h // heads if h and heads else None)
    inter = t.get("intermediate_size") or t.get("ffn_dim")
    vocab = t.get("vocab_size")
    params = st.get("total")
    estimated = False
    if not params and h and layers and vocab:
        params = int(layers * (4 * h * h + 3 * h * (inter or 4 * h)) + 2 * vocab * h)
        estimated = True
    experts = t.get("num_experts") or t.get("n_routed_experts") or t.get("num_local_experts")
    top_k = t.get("num_experts_per_tok") or t.get("moe_topk") or t.get("top_k")
    active = params
    moe_inter = t.get("moe_intermediate_size") or inter
    if experts and top_k and params and h and layers and moe_inter:
        expert_params = layers * experts * 3 * h * moe_inter
        if expert_params < params:
            active = int(params - expert_params * (1 - top_k / experts))
    card = meta.get("cardData") or {}
    return {
        "id": meta.get("id", rid), "created": (meta.get("createdAt") or "")[:10],
        "updated": (meta.get("lastModified") or "")[:10], "pipeline": meta.get("pipeline_tag"),
        "library": meta.get("library_name"), "gated": meta.get("gated") or False,
        "base_model": card.get("base_model"), "languages": card.get("language"),
        "downloads": meta.get("downloads"), "likes": meta.get("likes"),
        "architecture": (cfg.get("architectures") or [cfg.get("model_type")])[0] if cfg else None,
        "params": params, "params_estimated": estimated, "active_params": active if active != params else None,
        "experts": experts, "experts_per_token": top_k,
        "dtype_on_hub": ", ".join(st.get("parameters") or {}) or (t.get("torch_dtype") or t.get("dtype")),
        "layers": layers, "hidden": h, "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim,
        "intermediate": inter, "vocab": vocab, "context": t.get("max_position_embeddings"),
        "mla_kv_rank": t.get("kv_lora_rank"), "rope_dim": t.get("qk_rope_head_dim"),
        "sliding_window": t.get("sliding_window"),
        **({"config_from": cfg_from} if cfg_from != rid else {}),
    }


# ---------------- resources ----------------

BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "q8": 1.06, "q6": 0.82, "q5": 0.69,
         "q4": 0.56, "int4": 0.56, "nf4": 0.56, "awq": 0.56, "gptq": 0.56, "fp4": 0.56, "q3": 0.44, "q2": 0.33}
# name pattern -> (dense bf16 tensor TFLOPS, memory bandwidth GB/s). Published specs, rounded.
GPU_SPECS: list[tuple[str, float, float]] = [
    (r"b300|gb300", 2500, 8000), (r"b200|gb200", 2250, 8000), (r"h200", 989, 4800), (r"h100.*pcie", 756, 2000),
    (r"h100|h800|gh200", 989, 3350), (r"a100.*40", 312, 1555), (r"a100|a800", 312, 2039), (r"l40s", 362, 864),
    (r"l40\b", 181, 864), (r"l4\b", 121, 300), (r"a10g?\b", 125, 600), (r"a40\b", 150, 696), (r"t4\b", 65, 320),
    (r"v100", 125, 900), (r"rtx pro 6000", 500, 1792), (r"rtx\s?6000 ada", 365, 960), (r"a6000", 155, 768),
    (r"5090", 209, 1792), (r"5080", 113, 960), (r"5070 ti", 88, 896), (r"5070", 62, 672), (r"5060", 46, 448),
    (r"4090", 165, 1008), (r"4080", 97, 717), (r"4070 ti", 80, 504), (r"4070", 58, 504), (r"4060 ti", 44, 288),
    (r"4060", 30, 272), (r"3090", 71, 936), (r"3080", 59, 760), (r"3070", 41, 448), (r"3060 ti", 32, 448),
    (r"3060", 25, 360), (r"2080", 40, 448), (r"2070", 30, 448),
]
MFU = 0.35            # share of peak a well-tuned training run sustains
DECODE_EFF = 0.7      # share of memory bandwidth single-stream decoding reaches
OVERHEAD_GB = 0.8     # CUDA context, allocator, workspaces


def gpu_specs(name: str | None) -> tuple[float, float] | None:
    low = (name or "").lower()
    return next(((tf, bw) for pat, tf, bw in GPU_SPECS if re.search(pat, low)), None)


def _lora_params(f: dict[str, Any], rank: int) -> int:
    """LoRA on every linear layer of the blocks (q, k, v, o, gate, up, down) — the common 'all-linear' setup."""
    h, layers, inter = f["hidden"], f["layers"], f["intermediate"] or 4 * f["hidden"]
    q = (f["heads"] or 1) * (f["head_dim"] or h)
    kv = (f["kv_heads"] or f["heads"] or 1) * (f["head_dim"] or h)
    per_layer = (h + q) + 2 * (h + kv) + (q + h) + 2 * (h + inter) + (inter + h)
    return rank * per_layer * layers


def _kv_bytes_per_token(f: dict[str, Any], kv_bytes: float) -> float:
    if f.get("mla_kv_rank"):   # DeepSeek-style latent attention caches one compressed vector per layer
        return f["layers"] * (f["mla_kv_rank"] + (f.get("rope_dim") or 64)) * kv_bytes
    return 2 * f["layers"] * (f["kv_heads"] or 1) * (f["head_dim"] or 0) * kv_bytes


def estimate_resources(model_id: str, mode: str = "inference", precision: str | None = None, seq_len: int = 4096,
                       batch: int = 1, lora_rank: int = 16, gradient_checkpointing: bool = True,
                       train_tokens: float | None = None, gpu: str | None = None, vram_gb: float | None = None,
                       num_gpus: int = 1, kv_cache: str = "bf16", width: int | None = None,
                       height: int | None = None, frames: int | None = None, steps: int | None = None,
                       cfg: float | None = None, variant: str | None = None) -> dict[str, Any]:
    diff = _maybe_diffusion(model_id, mode, precision, width, height, frames, steps, cfg, variant, lora_rank, gpu,
                            vram_gb)
    if diff is not None:
        return diff
    f = model_facts(model_id)
    if f.get("error"):
        return f
    if not f.get("params"):
        return {"model": f["id"], "error": "parameter count unknown (no safetensors index and no LLM config)",
                "facts": f, "next": "model_variants lists the file sizes; weights size ~= file size"}
    mode = mode.lower()
    if mode not in ("inference", "lora", "qlora", "full"):
        return {"error": "mode must be inference | lora | qlora | full"}
    P, active = f["params"], f.get("active_params") or f["params"]
    prec = (precision or ("nf4" if mode == "qlora" else "bf16")).lower()
    bpp = BYTES.get(prec)
    if bpp is None:
        return {"error": f"precision must be one of {', '.join(BYTES)}"}
    llm = all(f.get(k) for k in ("hidden", "layers", "heads"))
    parts: dict[str, float] = {"weights": P * bpp}
    notes: list[str] = []
    h, L = f.get("hidden") or 0, f.get("layers") or 0

    if mode == "inference":
        if llm:
            kvb = 1.0 if kv_cache.lower() in ("fp8", "int8") else 2.0
            parts["kv_cache"] = _kv_bytes_per_token(f, kvb) * seq_len * batch
            parts["activations"] = batch * min(seq_len, 8192) * max(h, f.get("intermediate") or 0) * 2 * 4
            notes.append(f"KV cache for {batch} x {seq_len} tokens in {kv_cache}; fp8 KV cache halves it")
        else:
            notes.append("not a decoder LLM: activations depend on resolution / frames / batch — measure them; "
                         "weights are the floor")
    else:
        if not llm:
            return {"model": f["id"], "error": "training estimates need a transformer LLM config (hidden size, "
                    "layers, heads)", "facts": f}
        act_layer = 34 * seq_len * batch * h          # bytes per layer in 16-bit (flash attention: no s^2 term)
        if gradient_checkpointing:
            parts["activations"] = L * seq_len * batch * h * 2 + act_layer
        else:
            parts["activations"] = L * act_layer
        parts["logits"] = batch * seq_len * (f.get("vocab") or 0) * 6   # fp32 logits + their gradient
        if mode in ("lora", "qlora"):
            n = _lora_params(f, lora_rank)
            parts["lora_weights_grads_adam"] = n * 16
            notes.append(f"LoRA r={lora_rank} on all linear layers: {n / 1e6:.1f}M trainable params")
            if mode == "qlora":
                notes.append("QLoRA: 4-bit NF4 base (~0.56 bytes/param with scales); ~30% slower than bf16 LoRA")
        else:
            parts["gradients"] = P * 2
            parts["adam_states_fp32"] = P * 12   # fp32 master weights + two moments
            notes.append("full fine-tune, mixed precision AdamW: 16 bytes/param before activations; 8-bit Adam "
                         "saves ~6 bytes/param; FSDP / ZeRO-3 splits weights+grads+states across GPUs")
        notes.append("activations: " + ("gradient checkpointing on" if gradient_checkpointing else
                                        "no checkpointing (fastest, most memory)") + f", seq {seq_len}, batch {batch}")
    if num_gpus > 1 and mode == "full":
        for k in ("weights", "gradients", "adam_states_fp32"):
            parts[k] /= num_gpus
        notes.append(f"FSDP / ZeRO-3 over {num_gpus} GPUs: weights, gradients and optimizer states sharded")
    elif num_gpus > 1 and mode == "inference":
        parts["weights"] /= num_gpus
        parts["kv_cache"] = parts.get("kv_cache", 0) / num_gpus
        notes.append(f"tensor parallel over {num_gpus} GPUs")
    total = sum(parts.values()) * 1.05 + OVERHEAD_GB * GB
    out: dict[str, Any] = {
        "model": f["id"], "mode": mode, "precision": prec,
        "params_b": round(P / 1e9, 2), "active_params_b": round(active / 1e9, 2) if active != P else None,
        "breakdown_gb": {k: round(v / GB, 2) for k, v in parts.items()},
        "total_gb_per_gpu": round(total / GB, 1), "assumptions": notes + [
            f"+5% fragmentation and {OVERHEAD_GB} GB CUDA context per GPU",
            "formulas, not a measurement: real use is usually within ±15%"],
    }

    hw = local_hardware()
    g = {"name": gpu, "vram_gb": vram_gb} if (gpu or vram_gb) else (hw.get("gpus") or [None])[0]
    if g:
        vram = g.get("vram_gb")
        out["gpu"] = {"name": g.get("name"), "vram_gb": vram}
        if vram:
            out["fits"] = total / GB <= vram * 0.95
            out["headroom_gb"] = round(vram - total / GB, 1)
        specs = gpu_specs(g.get("name"))
        if specs:
            tflops, bw = specs
            if mode == "inference" and llm:
                per_tok = active * bpp + _kv_bytes_per_token(f, 2.0) * seq_len / 2
                out["decode_tokens_per_s_batch1"] = round(bw * 1e9 * DECODE_EFF / per_tok, 1)
            if mode != "inference" and train_tokens:
                mult = 6 if mode == "full" else 4          # LoRA skips weight gradients of the frozen base
                flops = mult * active * float(train_tokens) * (1.3 if mode == "qlora" else 1.0)
                if gradient_checkpointing:
                    flops *= 4 / 3                          # one extra forward
                secs = flops / (tflops * 1e12 * MFU * max(1, num_gpus))
                out["train_time_hours"] = round(secs / 3600, 1)
                out["assumptions"].append(f"{tflops:.0f} TFLOPS bf16 peak at {int(MFU * 100)}% utilisation")
            if cc_from_name(g.get("name") or "") and prec == "fp8" and cc_from_name(g["name"]) < 8.9:
                out["assumptions"].append("this GPU has no native FP8: weights stored in 8 bits, computed in 16")
    out["if_it_does_not_fit"] = _advice(mode, prec)
    return out


def _maybe_diffusion(model_id: str, mode: str, precision: str | None, width: int | None, height: int | None,
                     frames: int | None, steps: int | None, cfg: float | None, variant: str | None, lora_rank: int,
                     gpu: str | None, vram_gb: float | None) -> dict[str, Any] | None:
    """Image / video diffusion repos get their own estimate (components, latent tokens, steps)."""
    from kb.research.diffusion import _tree, estimate_diffusion, is_diffusion

    rid = _repo(model_id)
    http = interactive_client(0.05)
    try:
        meta = http.get(f"{HF}/api/models/{rid}").json()
        if "error" in meta:
            return None
        files = _tree(http, rid)
    except Exception:
        return None
    finally:
        http.close()
    if not is_diffusion(meta, [f["path"] for f in files]):
        return None
    hw = local_hardware()
    g = {"name": gpu, "vram_gb": vram_gb} if (gpu or vram_gb) else (hw.get("gpus") or [None])[0]
    specs = gpu_specs((g or {}).get("name"))
    video = bool(frames and frames > 1)
    return estimate_diffusion(rid, meta, files, mode=mode, precision=precision, width=width or (1280 if video else 1024),
                              height=height or (720 if video else 1024), frames=frames or 1, steps=steps or 20,
                              cfg=cfg if cfg is not None else 1.0, variant=variant, lora_rank=lora_rank, gpu=g,
                              gpu_tflops=specs[0] if specs else None, ram_gb=hw.get("ram_gb"))


def _advice(mode: str, prec: str) -> list[str]:
    if mode == "inference":
        return ["model_variants: an FP8 / INT8 variant (near-lossless, half of bf16) before 4-bit",
                "shorter context or fp8 KV cache (kv_cache='fp8')", "offload some layers to RAM (slower)",
                "tensor parallel across GPUs (num_gpus)"]
    if mode == "full":
        return ["LoRA instead (mode='lora'): usually close in quality for adaptation tasks", "8-bit AdamW",
                "FSDP / ZeRO-3 over several GPUs (num_gpus) or CPU offload of optimizer states",
                "gradient checkpointing, shorter sequences, micro-batch 1 with gradient accumulation"]
    return ["QLoRA (mode='qlora')" if mode == "lora" else "shorter sequences or a smaller rank",
            "chunked / fused cross-entropy (Liger kernels, Unsloth) removes most of the logits memory",
            "gradient checkpointing on, micro-batch 1 with gradient accumulation",
            "fewer target modules (attention only) or a smaller rank", "a smaller model from latest_models"]


# ---------------- benchmarks in model cards ----------------

BENCH = re.compile(
    r"\b(MMLU|GPQA|HumanEval|MBPP|LiveCodeBench|LCB|SWE-?bench|BigCodeBench|GSM-?8K|MATH|AIME|HMMT|ARC|HellaSwag|"
    r"Wino\w*|TruthfulQA|IFEval|IF-?Bench|MT-?Bench|Arena-?Hard|AlpacaEval|BBH|BIG-?Bench|DROP|MuSR|LiveBench|"
    r"SimpleQA|HLE|Humanity|BFCL|Tau|τ|TAU|MMMU|MathVista|DocVQA|ChartQA|TextVQA|OCRBench|AI2D|MMBench|MMStar|"
    r"Video-?MME|MVBench|VBench|FID|FVD|CLIP|GenEval|DPG|T2I-?Comp\w*|WER|CER|LibriSpeech|FLEURS|Common ?Voice|"
    r"MTEB|BEIR|C-?Eval|CMMLU|RULER|LongBench|NIAH|Needle|MGSM|Multi-?IF|PolyMath|Codeforces|Aider|"
    r"ImageNet|COCO|ADE20K|NYU|KITTI|MOS|UTMOS|SIM|Seed-?TTS|EmergentTTS)\b", re.I)


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def card_tables(card: str) -> list[list[list[str]]]:
    """Markdown and HTML tables of a model card as rows of cells."""
    def html_table(m: re.Match[str]) -> str:   # one HTML table -> pipe rows, one per <tr>
        t = re.sub(r"\s+", " ", m.group(0))
        t = re.sub(r"(?i)</t[dh]>\s*", " | ", t)
        t = re.sub(r"(?i)<tr[^>]*>", "\n| ", t)
        t = re.sub(r"(?i)</tr>", " ", t)
        return "\n" + htmllib.unescape(re.sub(r"<[^>]+>", "", t)) + "\n\n"

    text = re.sub(r"(?is)<table.*?</table>", html_table, card)
    tables, cur = [], []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.count("|") >= 2:
            if re.fullmatch(r"\|?[\s:\-|]+\|?", s):   # separator row
                continue
            cur.append(_cells(s))
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return [t for t in tables if len(t) >= 2]


def benchmark_rows(card: str, max_rows: int = 30) -> list[dict[str, Any]]:
    out = []
    for t in card_tables(card):
        header = t[0]
        # a category column may come first; rows under a merged (rowspan) category cell are one cell short
        rows = [[""] * (len(header) - len(r)) + r if 0 < len(header) - len(r) <= 2 else r
                for r in t[1:] if r and BENCH.search(" ".join(r[:3]))]
        if not rows and BENCH.search(" ".join(header[1:])):   # benchmarks as columns, models as rows
            rows = t[1:]
        if rows:
            out.append({"header": [c[:40] for c in header][:12], "rows": [[c[:40] for c in r][:12] for r in rows][:max_rows]})
    return out[:4]


def compare_models(model_ids: list[str]) -> dict[str, Any]:
    ids = [_repo(m) for m in model_ids][:6]
    http = interactive_client(0.05)
    try:
        def one(rid: str) -> dict[str, Any]:
            f = model_facts(rid, http)
            if f.get("error"):
                return f
            try:
                card = http.get(f"{HF}/{rid}/raw/main/README.md").text
            except Exception:   # gated: the rendered card on the public model page still has the tables
                try:
                    card = http.get(f"{HF}/{rid}").text
                except Exception:
                    card = ""
            f["benchmarks_from_card"] = benchmark_rows(card)
            if not f["benchmarks_from_card"]:
                f["benchmarks_from_card"] = "none in the card: see the tech report (source_info(code=true)) or leaderboards"
            return f

        with ThreadPoolExecutor(max_workers=6) as pool:
            facts = list(pool.map(one, ids))
    finally:
        http.close()
    keys = ("created", "params", "active_params", "architecture", "context", "gated", "dtype_on_hub",
            "pipeline", "downloads", "likes", "base_model")
    table = {k: {f.get("id", "?"): f.get(k) for f in facts} for k in keys}
    return {
        "models": [f.get("id") for f in facts], "side_by_side": table,
        "benchmarks": {f.get("id", "?"): f.get("benchmarks_from_card", []) for f in facts},
        "errors": {f["id"]: f["error"] for f in facts if f.get("error")},
        "caveat": ("Card benchmarks are self-reported, often with different prompts, shots and versions: compare a "
                   "benchmark across models only when the setting matches, and prefer independent leaderboards "
                   "(latest_models lists them) for the final call."),
    }


# ---------------- datasets ----------------

DS_SERVER = "https://datasets-server.huggingface.co"


def dataset_info(dataset_id: str, sample_rows: int = 3) -> dict[str, Any]:
    rid = re.sub(r"^(hf:|https?://huggingface\.co/)(datasets/)?", "", dataset_id.strip()).strip("/")
    http = interactive_client(0.05)
    try:
        meta = http.get(f"{HF}/api/datasets/{rid}").json()
        if "error" in meta:
            return {"error": f"{rid}: {meta['error']}"}
        out: dict[str, Any] = {
            "id": meta.get("id", rid), "created": (meta.get("createdAt") or "")[:10],
            "updated": (meta.get("lastModified") or "")[:10], "downloads": meta.get("downloads"),
            "likes": meta.get("likes"), "gated": meta.get("gated") or False,
            "tags": [t for t in meta.get("tags") or [] if t.split(":")[0] in
                     ("task_categories", "language", "size_categories", "modality", "format")][:20],
        }

        def get(path: str, **params: Any) -> dict[str, Any]:
            try:
                return http.get(f"{DS_SERVER}/{path}", params={"dataset": rid, **params}).json()
            except Exception as e:
                return {"error": f"{type(e).__name__}"}

        size = get("size")
        if "size" in size:
            out["configs"] = [{"config": c["config"], "rows": c.get("num_rows"),
                               "parquet_gb": round((c.get("num_bytes_parquet_files") or 0) / GB, 2)}
                              for c in size["size"].get("configs", [])][:10]
            out["splits"] = [{"config": s["config"], "split": s["split"], "rows": s.get("num_rows")}
                             for s in size["size"].get("splits", [])][:20]
            ds = size["size"].get("dataset", {})
            out["total_rows"], out["parquet_gb"] = ds.get("num_rows"), round((ds.get("num_bytes_parquet_files") or 0) / GB, 2)
            if size.get("partial"):
                out["size_note"] = "partial: the server converted only the first part; the real size is larger"
        else:
            out["size_note"] = size.get("error") or "no size info (gated, script-based or too new)"
        if out.get("splits"):
            first = out["splits"][0]
            rows = get("first-rows", config=first["config"], split=first["split"])
            if "features" in rows:
                out["columns"] = {f["name"]: (f["type"].get("dtype") or f["type"].get("_type"))
                                  for f in rows["features"]}
                out["sample"] = [{k: (str(v)[:300] if v is not None else None) for k, v in r["row"].items()}
                                 for r in rows.get("rows", [])[:sample_rows]]
        try:
            trained = http.get(f"{HF}/api/models", params={"filter": f"dataset:{rid}", "sort": "downloads",
                                                           "direction": -1, "limit": 6}).json()
            out["models_trained_on_it"] = [{"id": m["id"], "downloads": m.get("downloads")} for m in trained]
        except Exception:
            pass
    finally:
        http.close()
    out["next"] = "fetch the dataset page for the card (collection method, filtering, known issues)"
    return out


# ---------------- code for a paper ----------------

ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})")
LISTS = re.compile(r"awesome|paper-?list|reading|survey|collection|\.github\.io|arxiv-daily|daily-?papers", re.I)


def _title_tokens(title: str) -> list[str]:
    """Method name candidates: the part before ':' and words with capitals or digits ('A-FloPS' -> 'aflops')."""
    head = title.split(":")[0]
    words = re.findall(r"[\w-]+", head)
    toks = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in words if re.search(r"[A-Z].*[A-Z]|\d", w) or len(words) <= 2]
    return [t for t in toks if t]


def paper_code(paper: str) -> dict[str, Any]:
    m = ARXIV_ID.search(paper)
    if not m:
        return {"error": "pass an arXiv id or URL (e.g. 2509.00036); for other papers use github_search with the title"}
    aid = m.group(1)
    http = interactive_client(0.05)
    out: dict[str, Any] = {"arxiv": aid, "url": f"https://arxiv.org/abs/{aid}"}
    try:
        try:
            p = http.get(f"{HF}/api/papers/{aid}").json()
            if "error" not in p:
                out.update({"title": p.get("title"), "official_code": p.get("githubRepo"),
                            "github_stars": p.get("githubStars"), "project_page": p.get("projectPage"),
                            "hf_upvotes": p.get("upvotes"), "summary": (p.get("ai_summary") or "")[:400] or None})
        except Exception:
            pass
        try:
            repos = http.get(f"{HF}/api/arxiv/{aid}/repos").json()
            for kind in ("models", "datasets", "spaces"):
                items = sorted(repos.get(kind) or [], key=lambda r: -(r.get("downloads") or r.get("likes") or 0))
                out[f"hf_{kind}"] = [{"id": r["id"], "downloads": r.get("downloads"), "likes": r.get("likes")}
                                     for r in items[:8]]
        except Exception:
            pass
        if not out.get("title"):
            try:
                from kb.research.sources import arxiv_by_id

                ap = arxiv_by_id(aid)
                out["title"] = ap.title if ap else None
            except Exception:
                pass
        if not out.get("official_code"):
            try:
                gh = http.get("https://api.github.com/search/repositories",
                              params={"q": f"{aid} in:readme,description", "sort": "stars", "per_page": 10}).json()
                repos = [{"repo": r["full_name"], "stars": r["stargazers_count"], "updated": (r.get("pushed_at") or "")[:10],
                          "url": r["html_url"], "description": (r.get("description") or "")[:120]}
                         for r in gh.get("items", []) if not LISTS.search(r["full_name"] + " " + (r.get("description") or ""))]
                tokens = _title_tokens(out.get("title") or "")
                for r in repos:
                    name = re.sub(r"[^a-z0-9]", "", r["repo"].split("/")[1].lower())
                    r["likely_official"] = bool(tokens and any(t in name or name in t for t in tokens if len(t) >= 4))
                repos.sort(key=lambda r: (not r["likely_official"], -r["stars"]))
                out["github_mentions"] = repos[:5]
            except Exception:
                out["github_mentions"] = "GitHub search unavailable (rate limit): try github_search later"
    finally:
        http.close()
    out["next"] = ("fetch the official repo (README: install, checkpoints, open issues) before relying on it; "
                   "an unofficial reimplementation may differ from the paper's numbers")
    return out
