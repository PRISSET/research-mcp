"""Memory and time for diffusion image / video models (diffusers repos and ComfyUI single-file repos).

An LLM config does not describe these: the cost is a denoiser (DiT / UNet) run `steps` times over latent tokens
that depend on the resolution and the frame count, plus a text encoder and a VAE that ComfyUI loads one at a time.
  * components and their weights come from the repo's files; the exact parameter count and dtype from the
    safetensors header (an HTTP range read of a few hundred KB, no download), several precisions side by side;
  * the shapes (hidden size, layers, patch, VAE down-sampling) from transformer/ and vae/config.json of the repo or
    of its base model, so latent tokens = frames' x (H / s / p) x (W / s / p);
  * per step: 2 x params x tokens + 4 x layers x tokens^2 x hidden FLOPs (attention dominates video), doubled
    with CFG; weights that do not fit stream from RAM over PCIe every step.
Formulas, not measurements: the result says so, with every assumption.
"""

from __future__ import annotations

import json
import math
import re
import struct
from typing import Any

from kb.research.sources import interactive_client

HF = "https://huggingface.co"
GB = 1024**3
DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "I32": 4,
               "I64": 8, "BOOL": 1, "F4": 0.5, "U4": 0.5}
ROLES = (("vae", r"(^|/)(vae|audio_vae)(/|_|$)"),          # before the denoiser: vae/diffusion_pytorch_model…
         ("text_encoder", r"(^|/)(text_encoders?(_\d)?|t5|clip|umt5|llm)(/|_|$)"),
         ("denoiser", r"(^|/)(transformer|unet|diffusion_models?|dit)(/|_|$)|diffusion_pytorch_model"))
SKIP = re.compile(r"(^|/)(loras?|embeddings|model_patches|controlnet|upscale|non_official|examples?|assets)/", re.I)
PREC_IN_NAME = [("nvfp4", r"nvfp4|fp4"), ("int8", r"int8"), ("fp8", r"fp8|f8_?e4m3|e4m3"), ("w6a8", r"w6a8"),
                ("bf16", r"bf16"), ("fp16", r"fp16|f16"), ("fp32", r"fp32"), ("gguf", r"\.gguf$")]
PCIE_GBS = 20.0          # host -> GPU streaming of offloaded weights, effective PCIe 4.0 x16
EFF = 0.40               # share of dense bf16 peak a diffusion transformer sustains
OVERHEAD_GB = 0.8


def is_diffusion(meta: dict[str, Any], files: list[str]) -> bool:
    if meta.get("library_name") in ("diffusers", "diffusion-single-file"):
        return True
    if re.search(r"text-to-(image|video)|image-to-(image|video)|video", meta.get("pipeline_tag") or ""):
        return True
    return any(re.search(r"(^|/)(transformer|unet|diffusion_models)/", f) for f in files)


def _tree(http: Any, rid: str) -> list[dict[str, Any]]:
    data = http.get(f"{HF}/api/models/{rid}/tree/main", params={"recursive": "true"}).json()
    return [f for f in data if isinstance(f, dict) and f.get("type") == "file"] if isinstance(data, list) else []


def _size(f: dict[str, Any]) -> int:
    return (f.get("lfs") or {}).get("size") or f.get("size") or 0


def _prec_from_name(name: str) -> str | None:
    low = name.lower()
    return next((p for p, pat in PREC_IN_NAME if re.search(pat, low)), None)


def header_params(http: Any, rid: str, path: str) -> dict[str, Any] | None:
    """Parameter count by dtype from a safetensors header, read with two range requests."""
    url = f"{HF}/{rid}/resolve/main/{path}"
    try:
        n = struct.unpack("<Q", http.get(url, headers={"Range": "bytes=0-7"}).content[:8])[0]
        if n > 64 * 1024 * 1024:
            return None
        head = json.loads(http.get(url, headers={"Range": f"bytes=8-{7 + n}"}).content)
    except Exception:
        return None
    by: dict[str, int] = {}
    for k, v in head.items():
        if k != "__metadata__" and isinstance(v, dict):
            by[v.get("dtype", "?")] = by.get(v.get("dtype", "?"), 0) + math.prod(v.get("shape") or [0])
    return {"params": sum(by.values()), "by_dtype": by}


def components(http: Any, rid: str, files: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """role -> variants [{name, files, bytes, precision}]: shards of one diffusers folder are one variant; single
    files (ComfyUI repos) are one variant each."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for f in files:
        path = f.get("path", "")
        if not path.endswith((".safetensors", ".gguf", ".bin", ".pth", ".ckpt")) or SKIP.search(path):
            continue
        role = next((r for r, pat in ROLES if re.search(pat, path, re.I)), None)
        if role is None:
            continue
        shard = re.search(r"-\d{5}-of-\d{5}\.", path)
        key = path.rsplit("/", 1)[0] if shard or path.split("/")[-1].startswith(("diffusion_pytorch_model", "model")) else path
        v = out.setdefault(role, {}).setdefault(key, {"name": key, "files": [], "bytes": 0,
                                                    "precision": _prec_from_name(path)})
        v["files"].append(path)
        v["bytes"] += _size(f)
    return {r: sorted(vs.values(), key=lambda v: v["bytes"]) for r, vs in out.items()}


def _configs(http: Any, rid: str, files: list[str], base: list[str] | str | None) -> tuple[dict, dict, str | None]:
    """transformer + vae config.json of the repo, else of its base model (ComfyUI repos carry no configs)."""
    for repo, names in [(rid, files)] + [(b, None) for b in ([base] if isinstance(base, str) else base or [])][:2]:
        if names is None:
            try:
                names = [f["path"] for f in _tree(http, repo)]
            except Exception:
                continue
        tc = next((n for n in names if re.search(r"(^|/)(transformer|unet)/config\.json$", n)), None)
        if not tc:
            continue
        vc = next((n for n in names if n.startswith(tc.rsplit("transformer", 1)[0]) and re.search(r"(^|/)vae/config\.json$", n)), None)
        try:
            t = http.get(f"{HF}/{repo}/resolve/main/{tc}").json()
            v = http.get(f"{HF}/{repo}/resolve/main/{vc}").json() if vc else {}
            return t, v, repo
        except Exception:
            continue
    return {}, {}, None


def _vae_factors(v: dict[str, Any]) -> tuple[int, int]:
    if v.get("spatial_downsample_factors"):
        s = math.prod(v["spatial_downsample_factors"])
    elif v.get("spatial_compression_ratio"):
        s = int(v["spatial_compression_ratio"])
    else:
        mult = v.get("dim_mult") or v.get("block_out_channels") or [1, 2, 4, 4]
        s = 2 ** (len(mult) - 1)
    if v.get("temporal_downsample_factors"):
        t = math.prod(v["temporal_downsample_factors"])
    elif v.get("temporal_compression_ratio"):
        t = int(v["temporal_compression_ratio"])
    elif isinstance(v.get("temperal_downsample"), list):   # Wan-style VAEs spell it this way
        t = 2 ** sum(bool(x) for x in v["temperal_downsample"])
    else:
        t = 1
    return s, t


def _shape(t: dict[str, Any], params: float) -> dict[str, Any]:
    heads, hd = t.get("num_attention_heads"), t.get("attention_head_dim")
    hidden = t.get("hidden_size") or t.get("inner_dim") or (heads * hd if heads and hd else None)
    layers = (t.get("num_layers") or t.get("num_hidden_layers") or 0) + (t.get("num_single_layers") or 0) + \
        (t.get("num_refiner_layers") or 0)
    ffn = t.get("ffn_dim") or (int(hidden * t["mlp_ratio"]) if hidden and t.get("mlp_ratio") else None)
    guessed = False
    if not hidden or not layers:   # no config: a DiT block holds ~12 h^2 params; assume the common 48 layers
        layers, guessed = layers or 48, True
        hidden = hidden or int(math.sqrt(params / (12 * layers)))
    p = t.get("patch_size") or 2
    pt, ph = (p[0], p[1]) if isinstance(p, list) and len(p) == 3 else (1, p if isinstance(p, int) else p[-1])
    return {"hidden": hidden, "layers": layers, "ffn": ffn or 4 * hidden, "patch_t": pt, "patch_hw": ph,
            "guessed": guessed}


def _pick(variants: list[dict[str, Any]], precision: str | None, prefer: str | None) -> dict[str, Any] | None:
    vs = [v for v in variants if not prefer or prefer.lower() in v["name"].lower()] or variants
    if precision:
        hit = [v for v in vs if v["precision"] == precision.lower()]
        if hit:
            return max(hit, key=lambda v: v["bytes"])
    return max(vs, key=lambda v: v["bytes"]) if vs else None


def estimate_diffusion(rid: str, meta: dict[str, Any], files: list[dict[str, Any]], *, mode: str = "inference",
                       precision: str | None = None, width: int = 1024, height: int = 1024, frames: int = 1,
                       steps: int = 20, cfg: float = 1.0, variant: str | None = None, lora_rank: int = 32,
                       gpu: dict[str, Any] | None = None, gpu_tflops: float | None = None,
                       ram_gb: float | None = None) -> dict[str, Any]:
    http = interactive_client(0.05)
    try:
        comps = components(http, rid, files)
        if not comps.get("denoiser"):
            return {"model": rid, "error": "no denoiser weights found in the repo (transformer / unet / "
                    "diffusion_models)", "files_seen": len(files)}
        # the denoiser in the precision asked for; the text encoder too when the repo has it (else its largest)
        chosen = {r: _pick(vs, precision if r != "vae" else None, variant if r == "denoiser" else None)
                  for r, vs in comps.items()}
        den = chosen["denoiser"]
        head = header_params(http, rid, den["files"][0]) if len(den["files"]) == 1 else None
        if head:
            params = head["params"]
        else:   # sharded: bytes / bytes-per-param of the precision named (bf16 if none)
            bpp = {"fp8": 1, "int8": 1, "nvfp4": 0.56, "w6a8": 0.8, "fp32": 4}.get(den["precision"] or "", 2)
            params = int(den["bytes"] / bpp)
        t_cfg, v_cfg, cfg_from = _configs(http, rid, [f["path"] for f in files], (meta.get("cardData") or {}).get("base_model"))
    finally:
        http.close()

    shape = _shape(t_cfg, params)
    s, tf = _vae_factors(v_cfg) if v_cfg else ((16 if frames > 1 else 8), (4 if frames > 1 else 1))
    if frames <= 1:
        tf = 1
    lat_f = (frames - 1) // tf + 1 if frames > 1 else 1
    tokens = math.ceil(lat_f / shape["patch_t"]) * math.ceil(height / s / shape["patch_hw"]) * \
        math.ceil(width / s / shape["patch_hw"])
    batch = 2 if cfg and cfg > 1 else 1
    h, L, ffn = shape["hidden"], shape["layers"], shape["ffn"]
    w_den = den["bytes"]
    w_te = (chosen.get("text_encoder") or {}).get("bytes", 0)
    w_vae = (chosen.get("vae") or {}).get("bytes", 0)
    act = batch * tokens * 2 * (6 * h + 2 * ffn)          # one block's transient tensors, flash attention
    notes = [f"latent tokens {tokens:,} = {lat_f} latent frames x {math.ceil(height / s)}x{math.ceil(width / s)} "
             f"latents / patch {shape['patch_t']}x{shape['patch_hw']}x{shape['patch_hw']} (VAE /{s} space, /{tf} time)",
             f"CFG {'on: batch 2 per step' if batch == 2 else 'off (cfg 1, distilled / turbo): batch 1'}",
             "ComfyUI-style stages: text encoder, then denoiser, then VAE decode — one resident at a time"]
    if shape["guessed"]:
        notes.append("no transformer config found: hidden size guessed from the parameter count, 48 layers assumed")
    if cfg_from and cfg_from != rid:
        notes.append(f"shapes from the base model {cfg_from}")
    out: dict[str, Any] = {
        "model": rid, "kind": "diffusion", "mode": mode,
        "denoiser": {"file": den["name"], "precision": den["precision"], "params_b": round(params / 1e9, 2),
                     "weights_gb": round(w_den / GB, 2), **({"by_dtype_b": {k: round(v / 1e9, 2) for k, v in
                                                                             head["by_dtype"].items() if v > 1e7}}
                                                            if head else {})},
        "text_encoder": {"file": (chosen.get("text_encoder") or {}).get("name"), "weights_gb": round(w_te / GB, 2)},
        "vae": {"file": (chosen.get("vae") or {}).get("name"), "weights_gb": round(w_vae / GB, 2)},
        "variants": {r: [f"{v['name']} ({v['bytes'] / GB:.1f} GB)" for v in vs][:8] for r, vs in comps.items()},
        "latent_tokens": tokens,
    }
    ovh = OVERHEAD_GB * GB
    if mode in ("lora", "qlora", "full"):
        n_lora = lora_rank * L * (4 * 2 * h + 2 * (h + ffn))     # q, k, v, o and the two MLP projections
        ckpt = L * batch * tokens * h * 2                          # block inputs kept with gradient checkpointing
        other = (params * 14 if mode == "full" else n_lora * 16) + ckpt + act * 2
        notes.append(f"training: gradient checkpointing, LoRA r={lora_rank} on attention + MLP "
                     f"({n_lora / 1e6:.0f}M params)" if mode != "full" else "full fine-tune: AdamW states 14 B/param")
        stages = {"train_step": (w_den, other)}
        flops_step = 3 * (2 * params * tokens + 4 * L * tokens ** 2 * h)   # forward + backward
    else:
        vae_act = (4 if frames > 1 else 1) * width * height * 128 * 2 * 3   # first decoder stage, one temporal chunk
        stages = {"text_encoder": (w_te, 0.0), "denoise": (w_den, act), "vae_decode": (w_vae, vae_act)}
        notes.append("VAE decode: rough (decoders differ a lot); tiled VAE decode cuts it several times")
        flops_step = batch * (2 * params * tokens + 4 * L * tokens ** 2 * h)
    out["stages_gb"] = {k: round((w + o) * 1.05 / GB + OVERHEAD_GB, 1) for k, (w, o) in stages.items()}
    out["peak_vram_gb"] = max(out["stages_gb"].values())
    stream_s = 0.0
    if gpu and gpu.get("vram_gb"):
        vram = gpu["vram_gb"] * GB * 0.95
        out["gpu"] = gpu
        out["fits"] = all((w + o) * 1.05 + ovh <= vram for w, o in stages.values())
        plan = {}
        for k, (w, o) in stages.items():
            room = vram - o * 1.05 - ovh                   # what is left for this stage's weights
            if w * 1.05 <= room:
                plan[k] = "fits"
            elif room < 0.5 * GB:
                plan[k] = "does not fit even with all weights in RAM: fewer frames / lower resolution / tiled VAE"
            else:
                streamed = w - room / 1.05
                secs = streamed / GB / PCIE_GBS
                if k in ("denoise", "train_step"):
                    stream_s = secs
                    plan[k] = f"{streamed / GB:.1f} GB of weights stream from RAM every step (+{secs:.1f} s/step)"
                else:
                    plan[k] = f"{streamed / GB:.1f} GB of weights stream from RAM once (+{secs:.0f} s)"
        if not out["fits"]:
            out["offload"] = plan
            out["ram_needed_gb"] = round((w_den + w_te + w_vae) / GB + 2, 0)
            if ram_gb:
                out["ram_ok"] = out["ram_needed_gb"] <= ram_gb * 0.9
                if not out["ram_ok"]:
                    out["offload"]["ram"] = (f"needs ~{out['ram_needed_gb']:.0f} GB RAM for offloaded weights, the "
                                             f"machine has {ram_gb:.0f} GB: smaller variants (see `variants`) or "
                                             "load from disk (mmap, much slower)")
    if gpu_tflops:
        per_step = flops_step / (gpu_tflops * 1e12 * EFF) + stream_s
        out["seconds_per_step"] = round(per_step, 1)
        if mode == "inference":
            out["total_minutes"] = round(per_step * steps / 60, 1)
            notes.append(f"{steps} steps; text encoding and VAE decode not included")
        notes.append(f"{gpu_tflops:.0f} TFLOPS dense bf16 peak at {int(EFF * 100)}% (fp8 / int8 kernels can be faster "
                     "on Ada / Hopper / Blackwell); attention {:.0%} of the compute".format(
                         4 * L * tokens ** 2 * h / max(1, 2 * params * tokens + 4 * L * tokens ** 2 * h)))
    out["assumptions"] = notes + [f"+5% fragmentation and {OVERHEAD_GB} GB CUDA context",
                                  "formulas, not a measurement: check against one real run"]
    out["if_it_does_not_fit"] = ["model_variants / a smaller precision variant from `variants` (fp8 / int8 before "
                                 "4-bit)", "fewer frames or a lower resolution (attention grows with tokens squared)",
                                 "tiled VAE decode", "a distilled / turbo LoRA: fewer steps and cfg 1 halve the work"]
    return out
