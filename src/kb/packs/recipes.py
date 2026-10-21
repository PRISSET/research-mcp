"""Training recipes: the settings that published weights were actually trained with.

kohya-ss trainers (sd-scripts, musubi-tuner and the GUIs built on them) write the whole training config into the
safetensors header: learning rates, network dim / alpha, optimizer with its arguments, scheduler, steps, epochs,
batch, the datasets with image counts and repeats, resolution, noise offset, timestep sampling. ai-toolkit writes
its version, steps and epochs; diffusers / PEFT the adapter config; the rank is read from the tensor shapes when
nothing says it. Only the header is downloaded (one or two range requests). Fine-tuned LLMs carry their settings in
the model card: the Hugging Face Trainer's "Training hyperparameters" list or the Axolotl config.

recipes() sums them up per model family: median and spread of every setting, the usual optimizer and scheduler,
the share using noise offset or min-SNR, and the most downloaded examples — what people who published working
weights used, not one blog's opinion.
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from kb.store import Database

HF = "https://huggingface.co"
CIVITAI = "https://civitai.com/api/v1"
HEADER_GUESS = 65536    # most headers fit: one request instead of two
MAX_HEADER = 32 * 1024 * 1024

# Model families, most specific first. Matched against base model ids, Civitai base models and kohya's base version.
FAMILIES: list[tuple[str, str]] = [
    (r"kontext", "flux.1-kontext"), (r"flux[.\-_ ]?2|flux2", "flux.2"),
    (r"flux.{0,4}schnell|flux\.1 s\b", "flux.1-schnell"), (r"flux.{0,4}krea", "flux.1-krea"),
    (r"flux|^flux1$", "flux.1-dev"), (r"krea[-_ ]?2", "krea-2"),
    (r"qwen[-_ ]?image[-_ ]?edit", "qwen-image-edit"), (r"qwen[-_ ]?image", "qwen-image"),
    (r"wan[-_ ]?(video[-_ ]?)?2\.?2", "wan2.2"), (r"wan[-_ ]?(video[-_ ]?)?2\.?1|wan video", "wan2.1"),
    (r"hunyuan[-_ ]?video", "hunyuan-video"), (r"hunyuan[-_ ]?image", "hunyuan-image"), (r"ltx[-_ ]?2", "ltx-2"),
    (r"ltx", "ltx-video"), (r"minimax[-_ ]?h3", "minimax-h3"), (r"z[-_ ]?image", "z-image"), (r"hidream", "hidream"),
    (r"chroma", "chroma"), (r"illustrious", "illustrious"), (r"noob", "noobai"), (r"pony", "pony"),
    (r"sd[-_ ]?3\.5|stable-diffusion-3\.5", "sd3.5"), (r"sd[-_ ]?3|stable-diffusion-3", "sd3"),
    (r"sdxl|stable-diffusion-xl|xl[-_ ]?1\.0|sdxl_base", "sdxl"),
    (r"sd[-_ ]?1\.5|stable-diffusion-v1-5|sd_v1|v1-5", "sd1.5"), (r"sd[-_ ]?2|stable-diffusion-2", "sd2"),
    (r"cogvideo", "cogvideox"), (r"lumina", "lumina"), (r"kandinsky", "kandinsky"), (r"anima", "anima"),
]
TASKS: list[tuple[str, str]] = [
    ("speed", r"lightning|turbo|hyper|\b\d{1,2}[-_ ]?steps?\b|distill|\blcm\b|\bfast\b|causvid|accelerat"),
    ("edit", r"\bedit|swap|inpaint|try-?on|relight|upscal|restor|colori"),
    ("character", r"character|person|celebrit|\bface|portrait|woman|\bman\b|\bgirl|\bboy|actor|actress|likeness"),
    ("clothing", r"clothing|outfit|dress|costume|fashion"),
    ("style", r"style|\bart\b|aesthetic|anime|painting|illustrat|artist|sketch|cartoon"),
    ("motion", r"camera|orbit|motion|zoom|dolly|transition|animat|dance"),
    ("detail", r"detail|enhanc|realism|realistic|skin|hands?\b|quality"),
    ("concept", r"concept|pose|action|composition|effect|lighting|object|vehicle|background"),
]
LLM_TASKS = [("dpo", r"\bdpo\b|preference"), ("grpo", r"\bgrpo\b|\bppo\b|\brl\b|reinforce"), ("orpo", r"\borpo\b"),
             ("sft", r"\bsft\b|instruct|chat|fine-?tun")]
IMAGE_PIPES = {"text-to-image", "image-to-image", "text-to-video", "image-to-video", "image-text-to-image",
               "unconditional-image-generation", "video-to-video"}


# ---------------- parsing ----------------

def family(text: str | None) -> str | None:
    if not text:
        return None
    low = text.lower()
    for pat, name in FAMILIES:
        if re.search(pat, low):
            return name
    name = low.split("/")[-1]
    m = re.match(r"([a-z]+(?:[-_ ]?\d+(?:\.\d+)?)?)", name)   # Qwen3.5-4B-Base -> qwen3.5, Llama-3.1-8B -> llama-3.1
    return (m.group(1).replace("_", "-").replace(" ", "-") if m else name) or None


def task_of(words: list[str], title: str, llm: bool = False) -> str | None:
    hay = " ".join([*(w.lower() for w in words), title.lower()])
    for name, pat in LLM_TASKS if llm else TASKS:
        if re.search(pat, hay):
            return name
    return None


def _num(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", v)
        if m and v.strip().lower() not in ("none", "null", "false", "true"):
            try:
                f = float(m.group(0))
                return f if math.isfinite(f) else None
            except ValueError:
                return None
    return None


def _int(v: Any) -> int | None:
    f = _num(v)
    return int(f) if f is not None else None


def _json(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except ValueError:
        return None


def _round2(n: Any) -> int | None:
    """Two significant digits: download counts change every day, a recipe should not."""
    if not isinstance(n, (int, float)) or n <= 0:
        return 0 if n == 0 else None
    p = 10 ** max(int(math.log10(n)) - 1, 0)
    return int(round(n / p) * p)


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v not in (None, "", {}, [])}


def _datasets(meta: dict[str, str]) -> tuple[float, float, list[int], int | None]:
    """Images, images x repeats, resolutions and batch per device from kohya's ss_datasets (newer) or
    ss_dataset_dirs (older)."""
    datasets = _json(meta.get("ss_datasets")) or []
    per_device = _int(meta.get("ss_batch_size_per_device"))
    imgs = reps = 0.0
    res: list[int] = []
    for ds in datasets if isinstance(datasets, list) else []:
        per_device = per_device or _int(ds.get("batch_size_per_device"))
        res += [int(x) for x in ds.get("resolution") or [] if isinstance(x, (int, float))]
        for s in ds.get("subsets") or []:
            n = _num(s.get("img_count")) or 0
            imgs += n
            reps += n * (_num(s.get("num_repeats")) or 1)
    if not imgs:
        dirs = _json(meta.get("ss_dataset_dirs")) or {}
        for v in dirs.values() if isinstance(dirs, dict) else []:
            n = _num((v or {}).get("img_count")) or 0
            imgs += n
            reps += n * (_num((v or {}).get("n_repeats")) or 1)
    return imgs, reps, res, per_device


def _kohya(meta: dict[str, str], r: dict[str, Any], extra: dict[str, Any]) -> None:
    """The kohya-ss trainers' ss_* metadata: the whole training config."""
    r["trainer"] = "sd-scripts" if meta.get("ss_sd_scripts_commit_hash") else "kohya-ss"
    args = _json(meta.get("ss_network_args")) or {}
    module = meta.get("ss_network_module") or ""
    if "lycoris" in module:
        r["method"] = str(args.get("algo") or "lycoris").lower()
    elif isinstance(args, dict) and args.get("conv_dim"):
        r["method"] = "locon"
    r["rank"] = _int(meta.get("ss_network_dim")) or r["rank"]
    r["alpha"] = _num(meta.get("ss_network_alpha"))
    r["lr"] = _num(meta.get("ss_unet_lr")) or _num(meta.get("ss_learning_rate"))
    r["text_lr"] = _num(meta.get("ss_text_encoder_lr"))
    if r["text_lr"] == 0 or meta.get("ss_network_train_unet_only") == "True":
        r["text_lr"] = None
        extra["text_encoder"] = "frozen"
    opt = re.match(r"\s*([\w.]+)\s*(?:\((.*)\))?\s*$", meta.get("ss_optimizer") or "", re.S)
    if opt:
        r["optimizer"] = opt.group(1).split(".")[-1]
        if opt.group(2):
            extra["optimizer_args"] = opt.group(2)[:160]
    r["scheduler"] = meta.get("ss_lr_scheduler")
    r["steps"] = _int(meta.get("ss_steps")) or _int(meta.get("ss_max_train_steps"))
    r["epochs"] = _int(meta.get("ss_epoch")) or _int(meta.get("ss_num_epochs"))
    imgs, reps, res, per_device = _datasets(meta)
    if imgs:
        r["images"] = int(imgs)
        r["repeats"] = round(reps / imgs, 1)
    elif _int(meta.get("ss_num_train_images")):
        extra["samples_per_epoch"] = _int(meta.get("ss_num_train_images"))   # images x repeats
    accum = _int(meta.get("ss_gradient_accumulation_steps")) or 1
    r["batch"] = per_device * accum if per_device else None
    res += [int(x) for x in re.findall(r"\d{3,4}", meta.get("ss_resolution") or meta.get("modelspec.resolution") or "")]
    r["resolution"] = max(res) if res else None
    r["precision"] = meta.get("ss_mixed_precision")
    if meta.get("ss_fp8_base") == "True":
        extra["fp8_base"] = True
    for key, name in (("ss_noise_offset", "noise_offset"), ("ss_min_snr_gamma", "min_snr_gamma"),
                      ("ss_discrete_flow_shift", "flow_shift"), ("ss_network_dropout", "network_dropout"),
                      ("ss_caption_dropout_rate", "caption_dropout"), ("ss_lr_warmup_steps", "warmup_steps"),
                      ("ss_max_grad_norm", "max_grad_norm"), ("ss_multires_noise_iterations", "multires_noise"),
                      ("ss_guidance_scale", "guidance_scale")):
        v = _num(meta.get(key))
        if v:
            extra[name] = v
    for key, name in (("ss_timestep_sampling", "timestep_sampling"), ("ss_loss_type", "loss_type"),
                      ("ss_weighting_scheme", "weighting_scheme")):
        if meta.get(key) not in (None, "None", "none", ""):
            extra[name] = meta[key]
    started = _num(meta.get("ss_training_started_at"))
    if started:
        r["date"] = datetime.fromtimestamp(started, timezone.utc).date().isoformat()
    extra["base_hint"] = meta.get("modelspec.architecture") or meta.get("ss_base_model_version")


def from_header(head: dict[str, Any]) -> dict[str, Any]:
    """Training settings from a safetensors header (tensors + __metadata__)."""
    meta = {k: v for k, v in (head.get("__metadata__") or {}).items() if isinstance(v, str)}
    ranks: Counter[int] = Counter()
    method = None
    for k, t in head.items():
        if k == "__metadata__" or not isinstance(t, dict):
            continue
        shape = t.get("shape") or []
        if re.search(r"(lora_down|lora_A)(\.default)?\.weight$", k) and len(shape) >= 2:
            ranks[int(shape[0])] += 1
        elif "lokr_w1" in k:
            method = "lokr"
        elif "hada_w1" in k:
            method = "loha"
    r: dict[str, Any] = {"rank": ranks.most_common(1)[0][0] if ranks else None,
                         "method": method or ("lora" if ranks else None)}
    extra: dict[str, Any] = {}
    if any(k.startswith("ss_") for k in meta):
        _kohya(meta, r, extra)
    sw = _json(meta.get("software"))
    if isinstance(sw, dict) and sw.get("name"):
        r["trainer"] = f"{sw['name']} {sw.get('version') or ''}".strip()
        info = _json(meta.get("training_info")) or {}
        r["steps"] = _int(info.get("step")) or r.get("steps")
        r["epochs"] = _int(info.get("epoch")) or r.get("epochs")
    peft = _json(meta.get("lora_adapter_metadata"))
    if isinstance(peft, dict):
        r["trainer"] = r.get("trainer") or "diffusers / peft"
        for k, v in peft.items():
            if k.endswith((".r", "r")) and k.split(".")[-1] == "r":
                r["rank"] = _int(v) or r["rank"]
            elif k.split(".")[-1] == "lora_alpha":
                r["alpha"] = _num(v)
    for key in ("rank", "alpha", "steps"):   # trainers with plain keys
        if not r.get(key) and meta.get(key):
            r[key] = _num(meta[key]) if key == "alpha" else _int(meta[key])
    if extra.get("base_hint") is None:
        extra.pop("base_hint", None)
    r["extra"] = _clean(extra)
    return _clean(r)


CARD_KEYS = {
    "learning_rate": "lr", "train_batch_size": "batch_per_device", "total_train_batch_size": "batch",
    "gradient_accumulation_steps": "accum", "num_epochs": "epochs", "training_steps": "steps",
    "lr_scheduler_type": "scheduler", "lr_scheduler_warmup_ratio": "warmup_ratio", "optimizer": "optimizer",
    "lr_scheduler_warmup_steps": "warmup_steps", "seed": None, "eval_batch_size": None, "distributed_type": None,
    "num_devices": "devices", "mixed_precision_training": "precision",
    # axolotl config
    "lora_r": "rank", "lora_alpha": "alpha", "micro_batch_size": "batch_per_device", "sequence_len": "seq_len",
    "lr_scheduler": "scheduler", "adapter": "method", "load_in_4bit": "load_in_4bit", "warmup_ratio": "warmup_ratio",
    "lora_dropout": "lora_dropout", "max_steps": "steps",
}


def from_card(card: str) -> dict[str, Any]:
    """LLM fine-tune settings from a model card: the Trainer's hyperparameter list and / or an Axolotl config."""
    vals: dict[str, str] = {}
    for m in re.finditer(r"^\s*-?\s*([a-z_]+)\s*:\s*(.+?)\s*$", card, re.M):
        key = m.group(1)
        if key in CARD_KEYS and CARD_KEYS[key] and key not in vals:
            vals[key] = m.group(2).strip().strip('"\'')
    got = {CARD_KEYS[k]: v for k, v in vals.items()}
    if not ({"lr", "rank"} & set(got)):
        return {}
    r: dict[str, Any] = {"lr": _num(got.get("lr")), "rank": _int(got.get("rank")), "alpha": _num(got.get("alpha")),
                         "epochs": _int(got.get("epochs")), "steps": _int(got.get("steps")),
                         "scheduler": got.get("scheduler")}
    per = _int(got.get("batch_per_device"))
    r["batch"] = _int(got.get("batch")) or (per * (_int(got.get("accum")) or 1) * (_int(got.get("devices")) or 1)
                                            if per else None)
    opt = got.get("optimizer") or ""
    r["optimizer"] = (re.search(r"(?:use\s+(?:optimizer\s+)?)?([A-Za-z][\w.]*)", opt, re.I).group(1).split(".")[-1]
                      .removeprefix("OptimizerNames.") if opt else None)
    if r["optimizer"] and r["optimizer"].lower() == "use":
        r["optimizer"] = opt.split()[1] if len(opt.split()) > 1 else None
    method = (got.get("method") or "").lower()
    if method in ("lora", "qlora"):
        r["method"] = method
    elif r["rank"] or re.search(r"\bPEFT\s+\d", card):
        r["method"] = "qlora" if (got.get("load_in_4bit") or "").lower() == "true" else "lora"
    else:
        r["method"] = "full"
    r["precision"] = "fp16 (amp)" if "Native AMP" in (got.get("precision") or "") else None
    if "axolotl" in card.lower() and re.search(r"base_model\s*:", card):
        r["trainer"] = "axolotl"
    elif re.search(r"\bTRL\s+\d|\btrl\b", card):
        r["trainer"] = "trl"
    elif re.search(r"unsloth", card, re.I):
        r["trainer"] = "unsloth"
    elif re.search(r"llama-?factory", card, re.I):
        r["trainer"] = "llama-factory"
    elif re.search(r"Transformers\s+\d", card):
        r["trainer"] = "transformers"
    r["extra"] = _clean({"seq_len": _int(got.get("seq_len")), "warmup_ratio": _num(got.get("warmup_ratio")),
                         "warmup_steps": _int(got.get("warmup_steps")), "lora_dropout": _num(got.get("lora_dropout"))})
    return _clean(r)


class Retry(Exception):
    """A temporary failure (rate limit, outage): try this file again in the next build, do not remember it."""


def read_header(http: Any, url: str) -> dict[str, Any] | None:
    """The safetensors header of a remote file: one range request, two when the header is large. None when the file
    has none (not safetensors, needs a login); Retry on a temporary failure."""
    import httpx

    try:
        first = http.get(url, headers={"Range": f"bytes=0-{HEADER_GUESS - 1}"})
        if len(first.content) < 8:
            return None
        n = struct.unpack("<Q", first.content[:8])[0]
        if n > MAX_HEADER:
            return None
        raw = first.content[8:8 + n] if n + 8 <= len(first.content) else \
            http.get(url, headers={"Range": f"bytes=8-{7 + n}"}).content
        head = json.loads(raw)
        return head if isinstance(head, dict) else None
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (429, 500, 502, 503, 504):
            raise Retry(str(e)) from e
        return None
    except (httpx.TransportError, RuntimeError) as e:
        raise Retry(str(e)) from e
    except (ValueError, UnicodeDecodeError, struct.error):
        return None


def known(m: dict[str, Any], cache: dict[str, Any], civitai: bool = False) -> bool:
    """Was this repo revision / Civitai version read before (its recipe needs no download)?"""
    if civitai:
        v = (m.get("modelVersions") or [{}])[0]
        return f"civitai:{v.get('id')}" in cache
    return f"hf:{m['id']}@{m.get('sha')}" in cache


# ---------------- collecting (pack build) ----------------

def useful(r: dict[str, Any]) -> bool:
    return bool(r.get("lr") or r.get("steps") or r.get("epochs") or r.get("alpha"))


def _pick_file(files: list[str]) -> str | None:
    st = [f for f in files if f.endswith(".safetensors") and not re.search(
        r"(^|/)(text_encoder|vae|tokenizer|unet|transformer)/|optimizer", f)]
    final = [f for f in st if not re.search(r"(step|checkpoint|epoch)[-_]?\d|[-_]0{2,}\d+\.safetensors$", f, re.I)]
    pool = final or st
    return min(pool, key=lambda f: (f.count("/"), len(f))) if pool else None


def hf_models(http: Any, filt: str, sort: str, limit: int) -> list[dict[str, Any]]:
    params = [("filter", filt), ("sort", sort), ("direction", "-1"), ("limit", str(min(limit, 1000)))] + \
             [("expand[]", e) for e in ("sha", "siblings", "downloads", "likes", "tags", "createdAt", "pipeline_tag",
                                        "author")]
    data = http.get(f"{HF}/api/models", params=params).json()
    return data if isinstance(data, list) else []


def _text(http: Any, url: str) -> str:
    import httpx

    try:
        return http.get(url).text
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (429, 500, 502, 503, 504):
            raise Retry(str(e)) from e
        return ""
    except httpx.TransportError as e:
        raise Retry(str(e)) from e


def hf_recipe(http: Any, m: dict[str, Any], llm: bool, cache: dict[str, Any]) -> dict[str, Any] | None:
    """The recipe record of one Hugging Face repo (None when it says nothing about training)."""
    rid = m["id"]
    tags = m.get("tags") or []
    base = next((t.split(":", 2)[2] for t in tags if t.startswith(("base_model:adapter:", "base_model:finetune:"))),
                None)
    key = f"hf:{rid}@{m.get('sha')}"
    if key not in cache:
        if llm:
            card = _text(http, f"{HF}/{rid}/raw/main/README.md")
            parsed = from_card(card)
            if parsed.get("method") in ("lora", "qlora") and not parsed.get("rank"):
                cfg = _json(_text(http, f"{HF}/{rid}/raw/main/adapter_config.json")) or {}   # "" when missing
                if isinstance(cfg, dict):
                    parsed.update(_clean({"rank": _int(cfg.get("r")), "alpha": _num(cfg.get("lora_alpha"))}))
            cache[key] = {"parsed": parsed}
        else:
            f = _pick_file([s.get("rfilename", "") for s in m.get("siblings") or []])
            head = read_header(http, f"{HF}/{rid}/resolve/main/{f}") if f else None
            cache[key] = {"parsed": from_header(head) if head else {}, "file": f}
    hit = cache[key]
    parsed = dict(hit["parsed"])
    if not useful(parsed):
        return None
    extra = parsed.pop("extra", {})
    base = base or extra.get("base_hint")
    rec = {"type": "recipe", "id": f"recipe:hf:{rid.lower()}", "rev": m.get("sha"), **parsed,
           "url": f"{HF}/{rid}", "title": rid, "author": m.get("author") or rid.split("/")[0], "base_model": base,
           "family": family(base) or family(extra.get("base_hint")),
           "task": task_of(tags, rid, llm), "popularity": _round2(m.get("downloads") or 0),
           "date": parsed.get("date") or (m.get("createdAt") or "")[:10] or None,
           "extra": {**extra, **({"file": hit["file"]} if hit.get("file") else {}), "likes": _round2(m.get("likes"))}}
    return _clean(rec)


def civitai_models(http: Any, period: str, pages: int, per_page: int = 100) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    params: dict[str, Any] = {"types": "LORA", "sort": "Most Downloaded", "period": period, "limit": per_page,
                              "nsfw": "false"}
    for _ in range(pages):
        data = http.get(f"{CIVITAI}/models", params=params).json()
        out += data.get("items") or []
        cursor = (data.get("metadata") or {}).get("nextCursor")
        if not cursor:
            break
        params["cursor"] = cursor
    return out


def civitai_recipe(http: Any, m: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any] | None:
    versions = m.get("modelVersions") or []
    if not versions:
        return None
    v = versions[0]
    files = [f for f in v.get("files") or [] if (f.get("name") or "").endswith(".safetensors")]
    primary = next((f for f in files if f.get("primary")), files[0] if files else None)
    if not primary:
        return None
    key = f"civitai:{v['id']}"
    if key not in cache:
        head = read_header(http, primary["downloadUrl"])
        cache[key] = {"parsed": from_header(head) if head else {}}
    parsed = dict(cache[key]["parsed"])
    if not useful(parsed):
        return None
    extra = parsed.pop("extra", {})
    extra.pop("base_hint", None)
    stats = m.get("stats") or {}
    rec = {"type": "recipe", "id": f"recipe:civitai:{v['id']}", "rev": str(v["id"]), **parsed,
           "url": f"https://civitai.com/models/{m['id']}?modelVersionId={v['id']}", "title": m.get("name"),
           "author": (m.get("creator") or {}).get("username"), "base_model": v.get("baseModel"),
           "family": family(v.get("baseModel")), "task": task_of(m.get("tags") or [], m.get("name") or ""),
           "popularity": _round2(stats.get("downloadCount") or 0),
           "date": parsed.get("date") or (v.get("publishedAt") or "")[:10] or None,
           "extra": {**extra, "likes": _round2(stats.get("thumbsUpCount"))}}
    return _clean(rec)


# ---------------- reading (MCP tool) ----------------

COLUMNS = ("id", "url", "title", "author", "base_model", "family", "task", "trainer", "method", "rank", "alpha", "lr",
           "text_lr", "optimizer", "scheduler", "steps", "epochs", "batch", "images", "repeats", "resolution",
           "precision", "extra", "popularity", "date")
NUMERIC = ("lr", "text_lr", "rank", "alpha", "steps", "epochs", "batch", "images", "repeats", "resolution")


def store(db: Database, rec: dict[str, Any]) -> None:
    row = {c: rec.get(c) for c in COLUMNS}
    row["extra"] = json.dumps(rec.get("extra") or {}, ensure_ascii=False)
    db.conn.execute(f"INSERT OR REPLACE INTO recipes ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                    [row[c] for c in COLUMNS])


def _q(vals: list[float], q: float) -> float:
    s = sorted(vals)
    i = (len(s) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def _fmt(x: float) -> float | int:
    if abs(x) >= 100 or float(x).is_integer():
        return int(round(x))
    return float(f"{x:.3g}")


def _stats(vals: list[float], popular: list[float]) -> dict[str, Any]:
    out = {"median": _fmt(_q(vals, 0.5)), "p25": _fmt(_q(vals, 0.25)), "p75": _fmt(_q(vals, 0.75)), "n": len(vals)}
    common = Counter(_fmt(v) for v in vals).most_common(1)[0]
    if common[1] >= max(3, len(vals) // 4):
        out["most_common"] = common[0]
    if len(popular) >= 4:
        out["median_most_downloaded"] = _fmt(_q(popular, 0.5))
    return out


def _shares(values: list[Any], top: int = 4) -> list[dict[str, Any]]:
    vals = [v for v in values if v]
    return [{"value": v, "share": round(n / len(vals), 2), "n": n} for v, n in Counter(vals).most_common(top)] \
        if vals else []


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pop = sorted(rows, key=lambda r: r.get("popularity") or 0, reverse=True)
    top = pop[: max(len(pop) // 4, 1)]
    out: dict[str, Any] = {"recipes": len(rows), "authors": len({r.get("author") for r in rows if r.get("author")})}
    settings: dict[str, Any] = {}
    for f in NUMERIC:
        vals = [float(r[f]) for r in rows if r.get(f) is not None]
        if len(vals) >= 3:
            settings[f] = _stats(vals, [float(r[f]) for r in top if r.get(f) is not None])
    ratio = [r["alpha"] / r["rank"] for r in rows if r.get("alpha") and r.get("rank")]
    if len(ratio) >= 3:
        settings["alpha_over_rank"] = _stats(ratio, [])
    passes = [r["steps"] * r["batch"] / r["images"] for r in rows if r.get("steps") and r.get("batch") and r.get("images")]
    if len(passes) >= 3:
        settings["passes_per_image"] = {**_stats(passes, []), "means": "steps x batch / images"}
    out["settings"] = settings
    for f in ("method", "optimizer", "scheduler", "trainer", "precision", "task"):
        s = _shares([r.get(f) for r in rows])
        if s:
            out[f] = s
    extras = [r.get("extra") or {} for r in rows]
    tricks = {}
    for k in ("noise_offset", "min_snr_gamma", "flow_shift", "caption_dropout", "network_dropout", "fp8_base",
              "multires_noise", "seq_len", "warmup_ratio"):
        vals = [e[k] for e in extras if k in e]
        if vals:
            tricks[k] = {"share": round(len(vals) / len(rows), 2),
                         **({"median": _fmt(_q([float(v) for v in vals], 0.5))} if not isinstance(vals[0], bool) else {})}
    for k in ("timestep_sampling", "loss_type", "weighting_scheme"):
        s = _shares([e.get(k) for e in extras], 3)
        if s:
            tricks[k] = s
    if tricks:
        out["tricks"] = tricks
    return out


def recipes(db: Database, model: str, method: str | None = None, task: str | None = None,
            images: int | None = None, trainer: str | None = None, examples: int = 6) -> dict[str, Any]:
    fam = family(model)
    like = f"%{model.lower().strip()}%"
    rows = [dict(r) for r in db.conn.execute(
        "SELECT * FROM recipes WHERE family = ? OR family LIKE ? OR lower(base_model) LIKE ? OR lower(title) LIKE ?",
        (fam, like, like, like))]
    for r in rows:
        r["extra"] = json.loads(r.get("extra") or "{}")
    if not rows:
        have = db.conn.execute("SELECT family, COUNT(*) n FROM recipes GROUP BY family ORDER BY n DESC LIMIT 25").fetchall()
        if not have:
            return {"model": model, "recipes": 0,
                    "next": "no recipes installed: ask the user to run `uv run kb pack install lora-training` "
                            "(image / video LoRAs) or `llm-finetune` (or call knowledge_packs(install=…) with "
                            "their consent)"}
        return {"model": model, "recipes": 0, "families_known": {r[0]: r[1] for r in have if r[0]},
                "next": "no recipe for this model: try one of families_known (the model's base family)"}
    notes = []
    for name, val, col in (("method", method, "method"), ("task", task, "task"), ("trainer", trainer, "trainer")):
        if val:
            sub = [r for r in rows if val.lower() in str(r.get(col) or "").lower()]
            if len(sub) >= 3:
                rows = sub
            else:
                notes.append(f"only {len(sub)} recipes with {name}={val}: showing all")
    if images:
        near = [r for r in rows if r.get("images") and images / 2 <= r["images"] <= images * 2]
        if len(near) >= 5:
            rows = near
            notes.append(f"limited to datasets of {images // 2}-{images * 2} images")
        else:
            notes.append(f"only {len(near)} recipes with a dataset near {images} images: showing all sizes")
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r.get("family") or "unknown", []).append(r)
    order = sorted(groups, key=lambda f: (f != fam, -len(groups[f])))
    out_groups = []
    for f in order[:3]:
        g = groups[f]
        ex = sorted(g, key=lambda r: r.get("popularity") or 0, reverse=True)[:examples]
        out_groups.append({"family": f, **summarize(g), "examples": [
            _clean({k: r.get(k) for k in ("title", "url", "base_model", "trainer", "method", "rank", "alpha", "lr",
                                          "optimizer", "scheduler", "steps", "epochs", "batch", "images", "repeats",
                                          "resolution", "popularity", "date")}) for r in ex]})
    return {"model": model, "matched_family": fam, "recipes": len(rows), "notes": notes, "families": out_groups,
            "other_families": {f: len(groups[f]) for f in order[3:10]},
            "how_to_read": "Settings of published weights for this family (median and middle half; "
                           "median_most_downloaded = the most downloaded quarter). Recommend concrete values from "
                           "these, scaled to the user's dataset (passes_per_image), and say how many recipes / "
                           "authors back them. Popular is not proof of quality: check an example's page before "
                           "copying its settings."}
