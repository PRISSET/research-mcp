"""Choosing models: what is new for a task (latest_models) and which precision / quantized variant fits the
user's hardware (model_variants).

Two habits this module exists to break: recommending the model an agent remembers instead of the current one,
and defaulting to bf16 when an FP8 / INT8 variant would halve memory at near-zero quality cost.
"""

from __future__ import annotations

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import cache
from typing import Any

from kb.critic.sources import OFFICIAL_ORGS
from kb.research.sources import interactive_client

HF = "https://huggingface.co"

# Free-text task -> Hugging Face pipeline tags.
TASKS: list[tuple[str, list[str]]] = [
    (r"video.?(understand|qa|caption|llm|text)|понимани\w* видео", ["video-text-to-text"]),
    (r"time.?series|forecast|прогноз|временн\w* ряд", ["time-series-forecasting"]),
    (r"tabular|табличн", ["tabular-classification", "tabular-regression"]),
    (r"robot|vla\b|робот", ["robotics"]),
    (r"reinforcement|\brl\b|обучени\w* с подкреплени", ["reinforcement-learning"]),
    (r"voice.?(conv|clon)|клон\w* голос|rvc\b", ["audio-to-audio", "text-to-speech"]),
    (r"upscal|super.?resol|апскейл|restor|реставр", ["image-to-image"]),
    (r"inpaint|outpaint|дорисов", ["image-to-image", "text-to-image"]),
    (r"\bner\b|named entit|token classif|сущност", ["token-classification"]),
    (r"summar|суммар|пересказ", ["summarization", "text-generation"]),
    (r"question answ|\bqa\b|ответ\w* на вопрос", ["question-answering", "text-generation"]),
    (r"audio classif|sound classif|классификац\w* звук", ["audio-classification"]),
    (r"zero.?shot.?(image|classif)|clip\b", ["zero-shot-image-classification"]),
    (r"image.{0,15}classif|classif\w*.{0,15}image|классификац\w* изображ", ["image-classification"]),
    (r"text classif|sentiment|classif|классиф|тональн", ["text-classification"]),
    (r"keypoint|\bpose\b|\bпоз[аы]\b", ["keypoint-detection"]),
    (r"video segment|\btrack(ing)?\b|трекинг", ["mask-generation", "image-segmentation"]),
    (r"text.?to.?video|t2v|video gen|видео", ["text-to-video", "image-to-video", "image-text-to-video",
                                              "text-to-audio-video"]),
    (r"image.?to.?video|i2v|анимац", ["image-to-video"]),
    (r"text.?to.?image|image gen|t2i|картин|изображен", ["text-to-image"]),
    (r"image edit|редакт", ["image-to-image"]),
    (r"asr|speech.?recogn|transcri|распозна|whisper", ["automatic-speech-recognition"]),
    (r"tts|text.?to.?speech|speech synth|озвуч|синтез реч", ["text-to-speech"]),
    (r"music|audio gen|sound|звук|музык", ["text-to-audio"]),
    (r"vlm|vision.?language|multimodal|image.?text|ocr|мультимод", ["image-text-to-text"]),
    (r"omni|any.?to.?any", ["any-to-any"]),
    (r"embedding|retriev|вектор", ["feature-extraction", "sentence-similarity"]),
    (r"rerank", ["text-ranking"]),
    (r"llm|chat|text gen|language model|reason|code|языков", ["text-generation"]),
    (r"segment|сегмент", ["image-segmentation", "mask-generation"]),
    (r"depth|глубин", ["depth-estimation"]),
    (r"detect|детек", ["object-detection"]),
    (r"translat|перевод", ["translation", "text-generation"]),
    (r"3d", ["image-to-3d", "text-to-3d"]),
]
HF_TASKS = {t for _, tags in TASKS for t in tags}

# Where current rankings live (pointers for fetch_url / web search, not data).
LEADERBOARDS = {
    "text-generation": ["https://lmarena.ai/leaderboard", "https://artificialanalysis.ai/leaderboards/models"],
    "text-to-video": ["https://artificialanalysis.ai/text-to-video/arena", "https://lmarena.ai/leaderboard"],
    "image-to-video": ["https://artificialanalysis.ai/text-to-video/arena"],
    "text-to-image": ["https://artificialanalysis.ai/text-to-image/arena", "https://lmarena.ai/leaderboard"],
    "automatic-speech-recognition": ["https://huggingface.co/spaces/hf-audio/open_asr_leaderboard"],
    "text-to-speech": ["https://artificialanalysis.ai/text-to-speech/arena"],
    "feature-extraction": ["https://huggingface.co/spaces/mteb/leaderboard"],
    "sentence-similarity": ["https://huggingface.co/spaces/mteb/leaderboard"],
    "image-text-to-text": ["https://lmarena.ai/leaderboard", "https://huggingface.co/spaces/opencompass/open_vlm_leaderboard"],
    "video-text-to-text": ["https://video-mme.github.io/home_page.html#leaderboard"],
    "text-ranking": ["https://huggingface.co/spaces/mteb/leaderboard"],
    "time-series-forecasting": ["https://huggingface.co/spaces/Salesforce/GIFT-Eval"],
    "object-detection": ["https://paperswithcode.com/sota/object-detection-on-coco"],
    "any-to-any": ["https://lmarena.ai/leaderboard"],
}


def _months(d: str | None) -> int | None:
    if not d:
        return None
    try:
        x = date.fromisoformat(d[:10])
    except ValueError:
        return None
    t = date.today()
    return (t.year - x.year) * 12 + t.month - x.month


def task_tags(task: str) -> list[str]:
    t = task.strip().lower()
    if t in HF_TASKS:
        return [t]
    for pattern, tags in TASKS:
        if re.search(pattern, t):
            return tags
    return []


def _relation(tags: list[str]) -> str | None:
    for t in tags:
        m = re.match(r"base_model:(quantized|finetune|adapter|merge):", t)
        if m:
            return m.group(1)
    return "derivative" if any(t.startswith("base_model:") for t in tags) else None


def _family_key(repo_id: str) -> str:
    """'MiniMaxAI/MiniMax-H3' -> 'minimax-h3'; 'Wan-AI/Wan2.2-TI2V-5B' -> 'wan2.2'."""
    name = repo_id.split("/")[-1].lower()
    parts = re.split(r"[-_]", name)
    key = parts[0]
    if len(parts) > 1 and (len(key) <= 3 or re.fullmatch(r"[a-z]+", key)) and re.search(r"\d", parts[1]):
        key = f"{key}-{parts[1]}"
    return key if len(key) >= 4 else ""


def latest_models(task: str, months: int = 9, limit: int = 10) -> dict[str, Any]:
    """Newest strong base models for a task and the hottest derivatives, from Hugging Face."""
    tags = task_tags(task)
    http = interactive_client(0.1)
    pool: dict[str, dict[str, Any]] = {}
    try:
        # `filter` matches any tag, not only the main pipeline tag (MiniMax-H3 is "image-text-to-video" but
        # also tagged "text-to-video").
        queries = ([{"filter": t} for t in tags] if tags else [{"search": task}])
        for q in queries:
            for sort in ("trendingScore", "likes", "createdAt"):
                params = {**q, "sort": sort, "direction": -1, "limit": 100}
                for m in http.get(f"{HF}/api/models", params=params).json():
                    if "not-for-all-audiences" not in (m.get("tags") or []):
                        pool.setdefault(m["id"], m)
    finally:
        http.close()

    def row(m: dict[str, Any]) -> dict[str, Any]:
        created = (m.get("createdAt") or "")[:10]
        return {"id": m["id"], "url": f"{HF}/{m['id']}", "created": created, "age_months": _months(created),
                "task": m.get("pipeline_tag"), "likes": m.get("likes"), "downloads": m.get("downloads"),
                "trending": m.get("trendingScore"), "relation": _relation(m.get("tags") or []),
                "base_model": next((t.split(":")[-1] for t in m.get("tags") or [] if t.startswith("base_model:")), None)}

    rows = [row(m) for m in pool.values()]
    for r in rows:
        r["official"] = r["id"].split("/")[0].lower() in OFFICIAL_ORGS
    # Untagged community repos named after a lab's model ("someone/MiniMax-H3-hybrid") are derivatives too.
    families = {_family_key(r["id"]) for r in rows if r["official"] and r["relation"] is None}
    families.discard("")
    for r in rows:
        if r["relation"] is None and not r["official"] and any(f in r["id"].lower() for f in families):
            r["relation"] = "derivative (by name)"
    recent = [r for r in rows if r["age_months"] is not None and r["age_months"] <= months]
    # A real release collects likes fast; uploads with a handful of likes are noise. Labs' own releases first.
    base = sorted((r for r in recent if r["relation"] is None and (r["likes"] or 0) >= 20),
                  key=lambda r: (3 if r["official"] else 1) * (r["likes"] or 0) / (1 + (r["age_months"] or 0)) ** 0.5,
                  reverse=True)
    deriv = sorted((r for r in recent if r["relation"] is not None and (r["likes"] or 0) >= 10),
                   key=lambda r: (r["trending"] or 0, r["likes"] or 0), reverse=True)
    older_popular = sorted((r for r in rows if r["relation"] is None and (r["age_months"] or 0) > months),
                           key=lambda r: r["likes"] or 0, reverse=True)[:5]
    return {
        "task": task, "hf_tasks": tags or None, "window_months": months,
        "new_base_models": base[:limit], "hot_derivatives": deriv[:limit],
        "older_but_popular": older_popular,
        "leaderboards": sorted({u for t in tags for u in LEADERBOARDS.get(t, [])}),
        "note": ("Hugging Face shows open weights only: closed / API-only models and releases on other hubs are "
                 "missing — check the leaderboards (fetch) and the labs' announcements with web search. "
                 "Popularity is not quality: read the model cards and compare on the user's own constraints."),
    }


# ---------------- hardware ----------------

@cache
def local_hardware() -> dict[str, Any]:
    """The user's GPU (nvidia-smi still sees it when this server is pinned to the CPU) and RAM."""
    hw: dict[str, Any] = {"gpus": []}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        for line in out.stdout.strip().splitlines():
            name, mem, cc = (x.strip() for x in line.split(","))
            hw["gpus"].append({"name": name, "vram_gb": round(int(mem) / 1024, 1), "compute_cap": float(cc)})
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            kb = int(next(ln for ln in fh if ln.startswith("MemTotal")).split()[1])
        hw["ram_gb"] = round(kb / 1024**2, 1)
    except Exception:
        pass
    return hw


GPU_CC = [  # (name pattern, compute capability) — enough to tell FP8 / FP4 support apart
    (r"b200|b300|gb200|gb300", 10.0), (r"rtx\s?50\d0|rtx pro \d000 blackwell|blackwell", 12.0),
    (r"h100|h200|h800|h20\b|gh200", 9.0), (r"l4\b|l40|rtx\s?40\d0|ada|rtx\s?[2456]000 ada", 8.9),
    (r"a100|a800|a30\b", 8.0), (r"rtx\s?30\d0|a10\b|a40|a6000|a5000|a4000|a16", 8.6),
    (r"t4\b|rtx\s?20\d0|titan rtx", 7.5), (r"v100", 7.0),
]


def cc_from_name(gpu: str) -> float | None:
    low = gpu.lower()
    return next((cc for pattern, cc in GPU_CC if re.search(pattern, low)), None)


def compute_support(cc: float | None) -> dict[str, str]:
    if cc is None:
        return {"fp8": "unknown GPU", "fp4": "unknown GPU"}
    fp8 = ("native FP8 compute (Ada / Hopper / Blackwell): less memory and faster" if cc >= 8.9 else
           "no native FP8 compute: FP8 weights are stored in 8 bits and upcast per layer — about half the memory "
           "of bf16, same or slightly lower speed")
    fp4 = ("native FP4 (Blackwell)" if cc >= 10.0 else "no native FP4: NVFP4 / MXFP4 files need Blackwell or run "
           "through slow emulation — avoid")
    return {"fp8": fp8, "fp4": fp4}


# ---------------- variants ----------------

PRECISIONS: list[tuple[str, str, float]] = [   # (regex on file / repo name, label, quality rank: 0 = reference)
    (r"nvfp4|mxfp4|fp4", "fp4", 3.5),
    (r"fp8|float8|e4m3|e5m2", "fp8", 1.0),
    (r"w6a8", "w6a8", 2.0),
    (r"int8|w8a8|q8_0|q8\b|8bit|8-bit", "int8", 1.2),
    (r"q6_k|q6\b", "q6", 2.0),
    (r"q5_k|q5_|q5\b", "q5", 2.5),
    (r"q4_k|q4_|q4\b|int4|nf4|awq|gptq|w4a16|w4a8|4bit|4-bit|bnb", "4-bit", 3.0),
    (r"q3_|q3\b|q2_|q2\b|iq[123]", "<=3-bit", 5.0),
    (r"bf16", "bf16", 0.0),
    (r"fp16|f16\b|half", "fp16", 0.0),
    (r"fp32|f32\b", "fp32", 0.0),
]
QUALITY_NOTE = {
    "bf16": "reference quality", "fp16": "reference quality", "fp32": "reference (wasteful: twice bf16)",
    "fp8": "near-lossless for most models (scaled FP8 especially); halves memory vs bf16",
    "int8": "near-lossless; halves memory vs bf16; needs a runtime that supports the format",
    "w6a8": "small loss; needs a runtime that supports it",
    "q6": "very small loss (GGUF)", "q5": "small loss (GGUF)",
    "4-bit": "noticeable loss on details / long outputs — use only if 8-bit does not fit",
    "fp4": "noticeable loss unless the model was calibrated for it; native only on Blackwell",
    "<=3-bit": "large loss — last resort", "unknown": "precision not recognised from the name",
}
WEIGHT_EXT = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt")
SHARD = re.compile(r"-\d{5}-of-\d{5}")


def precision_of(name: str, dtype_hint: str | None = None) -> tuple[str, float]:
    low = name.lower()
    for pattern, label, rank in PRECISIONS:
        if re.search(pattern, low):
            return label, rank
    if dtype_hint:
        return precision_of(dtype_hint)
    return "unknown", 2.5        # unknown is not assumed lossless


def _dtype_hint(meta: dict[str, Any]) -> str | None:
    params = (meta.get("safetensors") or {}).get("parameters") or {}
    return max(params, key=params.get).lower().replace("_", "") if params else None


COMPONENTS = [(r"vae", "vae"), (r"text_?encoder|t5|umt5|clip|byt5|llm_?encoder|encoder", "text_encoder"),
              (r"lora", "lora"), (r"controlnet", "controlnet"), (r"upscal|esrgan", "upscaler"),
              (r"mmproj|projector|vision", "vision_projector")]
MAIN_COMPONENTS = {"model", "diffusion_models", "transformer", "unet", "checkpoints"}


def component_of(path: str) -> str:
    low = path.lower()
    folder = low.split("/")[0] if "/" in low else ""
    for pattern, comp in COMPONENTS:
        if re.search(pattern, low):
            return comp
    return folder if folder in MAIN_COMPONENTS else "model"


def runtime_note(repo: str, file: str) -> str | None:
    low = f"{repo}/{file}".lower()
    if "mlx" in low:
        return "MLX format: Apple Silicon only"
    if file.endswith(".gguf"):
        return "GGUF: llama.cpp / ComfyUI-GGUF / Ollama"
    if "awq" in low or "gptq" in low:
        return "AWQ / GPTQ: vLLM, SGLang, transformers"
    if any(x in low for x in ("convrot", "w6a8", "nvfp4")):
        return "special format: check which runtime the card names"
    return None


def _variants_in(meta: dict[str, Any], repo: str, min_gb: float) -> list[dict[str, Any]]:
    hint = _dtype_hint(meta)
    groups: dict[str, dict[str, Any]] = {}
    for s in meta.get("siblings") or []:
        f = s.get("rfilename", "")
        if not f.endswith(WEIGHT_EXT):
            continue
        key = SHARD.sub("", f)
        g = groups.setdefault(key, {"repo": repo, "file": key, "bytes": 0, "shards": 0})
        g["bytes"] += s.get("size") or 0
        g["shards"] += 1
    out = []
    for g in groups.values():
        size = g["bytes"] / 1e9
        if size < min_gb:
            continue
        label, rank = precision_of(g["file"].rsplit("/", 1)[-1])
        if label == "unknown":      # "model.safetensors" in "Qwen/Qwen3-8B-AWQ": the repo name says it
            label, rank = precision_of(repo.split("/")[-1], hint)
        pruned = "prun" in g["file"].lower() or "prun" in repo.lower()
        note = runtime_note(repo, g["file"])
        out.append({"repo": repo, "file": g["file"], "component": component_of(g["file"]), "runtime": note,
                    "precision": label, "size_gb": round(size, 1), "pruned": pruned,
                    "quality_rank": rank + (1.0 if pruned else 0.0),
                    "quality_note": QUALITY_NOTE.get(label, "") + ("; PRUNED: a smaller, different model — "
                                                                   "compare quality separately" if pruned else "")})
    return out


def _fit(size_gb: float, vram: float | None, ram: float | None) -> str:
    if vram is None:
        return "unknown (pass vram_gb)"
    if size_gb <= vram - 1.0:
        return "fits in VRAM" + (" (tight: little room for activations)" if size_gb > vram * 0.8 else "")
    if ram and size_gb <= vram + ram * 0.7:
        return "needs CPU offload (slower)"
    return "too big for this machine"


def model_variants(model_id: str, vram_gb: float | None = None, gpu_name: str | None = None, min_gb: float = 0.3,
                   community: int = 6) -> dict[str, Any]:
    rid = re.sub(r"^(hf:|https?://huggingface\.co/)", "", model_id.strip()).strip("/")
    hw = local_hardware()
    gpu = hw["gpus"][0] if hw.get("gpus") else None
    if gpu_name:          # planning for another GPU (a rental, a friend's card): judge by its name
        gpu = {"name": gpu_name, "vram_gb": vram_gb, "compute_cap": cc_from_name(gpu_name)}
    vram = vram_gb if vram_gb is not None else (gpu or {}).get("vram_gb")
    cc = (gpu or {}).get("compute_cap") if (gpu_name or vram_gb is None) else None
    http = interactive_client(0.05)
    try:
        meta = http.get(f"{HF}/api/models/{rid}", params={"blobs": "true"}).json()
        if "error" in meta:
            return {"error": f"{rid}: {meta['error']}"}
        quant = http.get(f"{HF}/api/models", params={"filter": f"base_model:quantized:{rid}", "sort": "downloads",
                                                     "direction": -1, "limit": community}).json()

        def blobs(m: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            try:
                return m, http.get(f"{HF}/api/models/{m['id']}", params={"blobs": "true"}).json()
            except Exception:
                return m, {}

        with ThreadPoolExecutor(max_workers=6) as pool:
            community_meta = list(pool.map(blobs, quant if isinstance(quant, list) else []))
    finally:
        http.close()

    variants = _variants_in(meta, rid, min_gb)
    for m, cm in community_meta:
        for v in _variants_in(cm, m["id"], min_gb):
            v.update(community=True, downloads=m.get("downloads"), likes=m.get("likes"))
            variants.append(v)
    for v in variants:
        v["fit"] = _fit(v["size_gb"], vram, hw.get("ram_gb"))

    apple = os.uname().sysname == "Darwin" if hasattr(os, "uname") else False
    main = [v for v in variants if v["component"] in MAIN_COMPONENTS
            and (apple or "Apple" not in (v.get("runtime") or ""))] or variants
    fitting = [v for v in main if v["fit"].startswith("fits")]
    offload = [v for v in main if v["fit"].startswith("needs")]
    best_fit = min(fitting, key=lambda v: (v["quality_rank"], -v["size_gb"]), default=None)
    best_off = min(offload, key=lambda v: (v["quality_rank"], v["size_gb"]), default=None)

    def name(v: dict[str, Any]) -> str:
        return f"{v['repo']}/{v['file']} ({v['precision']}, {v['size_gb']} GB)"

    advice = []
    if best_fit and best_fit["quality_rank"] <= 2.5:
        advice.append(f"best quality that fits in VRAM: {name(best_fit)}")
        if best_fit["precision"] in ("bf16", "fp16", "fp32") and best_fit["fit"].endswith("activations)"):
            smaller = [v for v in fitting if v["precision"] in ("fp8", "int8")]
            if smaller:
                s8 = min(smaller, key=lambda v: (v["pruned"], v["quality_rank"], v["size_gb"]))  # same model first
                advice.append(f"bf16 barely fits: {name(s8)} frees memory for longer / larger outputs at "
                              "near-zero quality cost")
    elif best_off and (best_fit is None or best_off["quality_rank"] + 1 < best_fit["quality_rank"]):
        advice.append(f"for quality: {name(best_off)} with CPU offload (slower) — {best_off['quality_note']}")
        if best_fit:
            advice.append(f"fully in VRAM only {name(best_fit)}: faster, but {best_fit['quality_note']}")
    elif best_fit:
        advice.append(f"fits in VRAM: {name(best_fit)} — {best_fit['quality_note']}")
    else:
        advice.append("too big for this machine even with offload: rent a GPU or use a hosted API")
    support = compute_support(cc)
    if any(v["precision"] == "fp8" for v in variants):
        advice.append(f"FP8 on this GPU: {support['fp8']}")
    if any(v["precision"] == "fp4" for v in variants):
        advice.append(f"FP4 on this GPU: {support['fp4']}")
    variants.sort(key=lambda v: (v.get("community", False), v["component"], v["quality_rank"], v["size_gb"]))
    return {
        "model": rid, "hardware": {"gpu": gpu, "vram_gb": vram, "ram_gb": hw.get("ram_gb")} if vram else hw,
        "advice": advice, "variants": variants[:60],
        "note": ("Sizes are the weight files only: text encoders, VAE, KV cache / activations need extra memory. "
                 "Prefer FP8 / INT8 over 4-bit; check the runtime (ComfyUI, diffusers, vLLM, llama.cpp) supports the "
                 "file format, and read the variant's card for its own quality notes."),
    }
