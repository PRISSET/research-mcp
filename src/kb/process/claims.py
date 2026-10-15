"""Rule-based extraction of quantitative claims from full-text chunks (extractor "rules-v1").

Kinds: speedup, memory, steps, latency, quality_metric (any benchmark score: accuracy, F1, BLEU, WER, FID, ...).

Captures the sentence verbatim with its numbers, the hardware and base models it names, so the critic can
compare "2.3x faster" against the conditions it was measured under. An LLM extractor can be added later
under another `extractor` tag without touching these rows.
"""

from __future__ import annotations

import json
import re
from typing import Any

from kb.store import Database

HARDWARE = re.compile(
    r"\b(H100|H200|H20|H800|A100|A800|A6000|A40|L40S?|L20|RTX\s?(?:PRO\s?)?\d{4}(?:\s?Ti)?|RTX\s?6000\s?Ada|"
    r"B200|B300|GB200|MI300X?|V100|T4|4090|5090|3090|TPU\s?v\d\w?|Ascend\s?910\w?)\b", re.IGNORECASE)
MODELS = re.compile(
    r"\b(GPT-?\d(?:\.\d)?(?:-?\w+)?|o\d(?:-mini)?|Claude(?:[- ]\d(?:\.\d)?)?|Gemini(?:[- ]\d(?:\.\d)?)?|Gemma[- ]?\d?|"
    r"Llama[- ]?\d(?:\.\d)?|Mistral|Mixtral|Qwen\d?(?:\.\d)?(?:-VL)?|DeepSeek(?:-[VR]\d)?|Phi-?\d|BERT|RoBERTa|T5|"
    r"ViT(?:-[SBLH])?|ResNet-?\d+|CLIP|SigLIP|DINOv\d|SAM\s?\d?|Whisper|"
    r"MiniMax[- ]?H3|Wan\s?2\.\d|Wan2\.\d|HunyuanVideo(?:[- ]1\.5)?|CogVideoX(?:-?\d\.?\d?)?|LTX-?(?:Video|2)|Mochi|"
    r"Stable Video Diffusion|SVD|FLUX(?:\.\d)?|SDXL|SD3(?:\.5)?|Stable Diffusion(?: \d(?:\.\d)?)?|DiT-XL|Sora|Veo\s?\d?)", re.IGNORECASE)
NUM = r"(\d+(?:\.\d+)?)"
GEN_CONTEXT = re.compile(r"generat|inference|sampling|denois|render|latency|end-to-end|runtime|wall-?clock|throughput|"
                         r"per token|decod|training|forward pass", re.IGNORECASE)

PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("speedup", re.compile(rf"{NUM}\s?(?:×|x|\$\\times\$|times)\s*(?:[\w-]+\s+){{0,2}}?(?:speed-?ups?|faster|acceleration|"
                           rf"lower latency|throughput)", re.IGNORECASE), "x"),
    ("speedup", re.compile(rf"(?:speed-?up|accelerat\w+|faster)\s+(?:of|by|up to|reaching|achieving)?\s*(?:up to\s+)?"
                           rf"{NUM}\s?(?:×|x|\$\\times\$|times)", re.IGNORECASE), "x"),
    ("memory", re.compile(rf"(?:reduc\w+|sav\w+|cut\w*|lower\w*)\s+(?:the\s+)?(?:peak\s+)?(?:GPU\s+)?memory[^.]{{0,40}}?"
                          rf"(?:by|to)\s+{NUM}\s?(%|×|x|GB)", re.IGNORECASE), ""),
    ("memory", re.compile(rf"{NUM}\s?(GB|GiB)\s+(?:of\s+)?(?:peak\s+)?(?:GPU\s+)?(?:memory|VRAM)", re.IGNORECASE), ""),
    ("steps", re.compile(rf"\b{NUM}[- ](?:step|NFE)s?\b[^.]{{0,60}}?(?:generation|sampling|inference|student|model)", re.IGNORECASE), "steps"),
    ("latency", re.compile(rf"{NUM}\s?(s|seconds|ms|minutes|min)\b[^.]{{0,30}}?(?:per|for)\s+(?:a\s+|one\s+)?"
                           rf"(?:video|clip|image|step|sample|token|query|request|batch|epoch|iteration)", re.IGNORECASE), ""),
    ("quality_metric", re.compile(rf"\b(VBench|FVD|FID|CLIPSIM|PSNR|SSIM|LPIPS|VQAScore|EvalCrafter|accuracy|top-1|top-5|"
                                  rf"F1|BLEU|ROUGE(?:-[1L2])?|METEOR|CIDEr|perplexity|PPL|WER|CER|mAP|AP50|mIoU|AUC|"
                                  rf"AUROC|MMLU|GSM8K|HumanEval|pass@1|Elo|win[- ]rate|exact match|EM)\b[^.\[$=]{{0,40}}?"
                                  rf"(?:of|score|(?<!\b[a-z])=|:|to|from|reaches|achieves)\s*\$?{NUM}(?![\d\]])(?!\s*\])", re.IGNORECASE), ""),
]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if 25 < len(s.strip()) < 600]


def extract(text: str) -> list[dict[str, Any]]:
    out = []
    for sent in sentences(text):
        gen_context = bool(GEN_CONTEXT.search(sent))
        hw = sorted({re.sub(r"\s+", " ", m.group(0)).upper() for m in HARDWARE.finditer(sent)})
        models = sorted({m.group(0) for m in MODELS.finditer(sent)})
        seen: set[tuple[str, float]] = set()
        for kind, pat, unit in PATTERNS:
            for m in pat.finditer(sent):
                if kind == "quality_metric":
                    metric, val = m.group(1), m.group(2)
                    u = metric.upper()
                else:
                    val = m.group(1)
                    u = unit or (m.group(2) if m.lastindex and m.lastindex >= 2 else "")
                try:
                    v = float(val)
                except ValueError:
                    continue
                if kind == "speedup" and not (1.05 <= v <= 1000):
                    continue
                if kind == "steps" and not (1 <= v <= 1000):
                    continue
                if kind == "latency" and not gen_context:
                    continue
                if (kind, v) in seen:
                    continue
                seen.add((kind, v))
                out.append({"kind": kind, "value": v, "unit": u, "hardware": hw, "models": models, "sentence": sent})
    return out


def extract_all(db: Database, rebuild: bool = False, paper_ids: list[str] | None = None) -> dict[str, int]:
    scope, params = "", []
    if paper_ids is not None:
        if not paper_ids:
            return {"chunks_scanned": 0, "claims": 0}
        scope = f" AND c.paper_id IN ({','.join('?' * len(paper_ids))})"
        params = list(paper_ids)
    if rebuild:
        db.conn.execute(f"DELETE FROM paper_claims WHERE extractor = 'rules-v1'{scope.replace('c.paper_id', 'paper_id')}",
                        params)
    rows = db.conn.execute(
        f"""SELECT c.id, c.paper_id, c.text FROM chunks c
           WHERE c.kind IN ('text', 'table', 'figure') AND c.section NOT LIKE '%Related Work%'
             AND c.paper_id NOT IN (SELECT DISTINCT paper_id FROM paper_claims WHERE extractor = 'rules-v1'){scope}""",
        params).fetchall()
    n = 0
    for r in rows:
        for c in extract(r["text"]):
            db.conn.execute(
                """INSERT INTO paper_claims (paper_id, chunk_id, kind, value, unit, hardware, models, sentence, extractor)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'rules-v1')""",
                (r["paper_id"], r["id"], c["kind"], c["value"], c["unit"], json.dumps(c["hardware"]),
                 json.dumps(c["models"]), c["sentence"]),
            )
            n += 1
    db.commit()
    return {"chunks_scanned": len(rows), "claims": n}


def find(db: Database, kind: str | None = None, hardware: str | None = None, model: str | None = None,
         text: str | None = None, paper_id: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
    sql = """SELECT cl.*, p.title, p.year, COALESCE(cr.grade, 'C') AS grade, c.section
             FROM paper_claims cl JOIN papers p ON p.id = cl.paper_id
             LEFT JOIN critique cr ON cr.source_id = p.id LEFT JOIN chunks c ON c.id = cl.chunk_id WHERE 1=1"""
    params: list[Any] = []
    for col, val in (("cl.kind", kind), ("cl.paper_id", paper_id)):
        if val:
            sql += f" AND {col} = ?"
            params.append(val)
    if hardware:
        sql += " AND cl.hardware LIKE ?"
        params.append(f"%{hardware.upper()}%")
    if model:
        sql += " AND (cl.models LIKE ? OR cl.sentence LIKE ?)"
        params += [f"%{model}%", f"%{model}%"]
    if text:
        for w in re.findall(r"[\w.+-]{3,}", text):
            sql += " AND (cl.sentence LIKE ? OR p.title LIKE ?)"
            params += [f"%{w}%", f"%{w}%"]
    sql += " ORDER BY CASE COALESCE(cr.grade, 'C') WHEN 'A' THEN 0 WHEN 'B' THEN 1 ELSE 2 END, p.year DESC LIMIT ?"
    params.append(limit)
    out = []
    for r in db.conn.execute(sql, params):
        d = dict(r)
        d["hardware"], d["models"] = json.loads(d["hardware"] or "[]"), json.loads(d["models"] or "[]")
        out.append(d)
    return out
