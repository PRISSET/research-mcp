"""find_methods: the strongest recent methods for a concrete goal, not a list of loosely related papers.

A plain keyword search misses the best answer when the paper uses another word ("sampler" vs "solver") and ranks by
citations, which favours old surveys. Here:
  1. several phrasings (the agent's own + synonym swaps) go to OpenAlex and arXiv, deep pools, recent years only;
  2. papers that cite the baseline the user wants to beat are added (newer methods compare against it);
  3. every candidate is ranked by meaning against the goal (multilingual embeddings), then by evidence:
     a measured gain in the abstract, training-free / drop-in, extra model calls, venue, code, recency;
  4. the abstract sentences with the claimed gains are returned, so the agent sees *what* each method buys.
"""

from __future__ import annotations

import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date
from typing import Any

from kb.research import sources
from kb.schemas import Paper
from kb.store import Database

log = logging.getLogger(__name__)

DEADLINE = 40.0
SYNONYMS: list[tuple[str, ...]] = [
    ("solver", "sampler", "integrator"),
    ("flow matching", "rectified flow"),
    ("acceleration", "speed-up", "few-step"),
    ("training-free", "plug-and-play"),
    ("quality", "fidelity"),
    ("audio", "speech"),
    ("video generation", "video diffusion"),
]
GAIN = re.compile(
    r"\b(FID|FVD|CLIP|VBench|IS|PSNR|SSIM|LPIPS|MOS|WER|CER|NFE|NFEs|steps?|speed-?up|faster|latency|throughput|"
    r"improv\w*|outperform\w*|reduc\w*|boost\w*|state-of-the-art|SOTA|better)\b|\d+(\.\d+)?\s*(%|×|x\b)", re.I)
NUMBER = re.compile(r"\d")
TRAINING_FREE = re.compile(r"training[- ]free|without (any )?(re)?training|no (additional |extra )?training|"
                           r"plug[- ]and[- ]play|drop[- ]in|zero[- ]shot|post[- ]hoc|retraining[- ]free", re.I)
NEEDS_TRAINING = re.compile(r"\b(we|our method|it) (train|fine-?tune|distill|learn)s?\b|\b(lightweight|small|auxiliary) "
                            r"(network|side-?net|module|predictor|head)\b|sidenet|negligible training|training cost|"
                            r"distillation|fine-?tuning the|learned (schedule|solver|coefficients?) on", re.I)
EXTRA_CALLS = re.compile(r"(additional|extra) (network|function|model) evaluations?|(two|2|three|3) (network |function )?"
                         r"evaluations per step|doubl\w+ the (cost|NFE)", re.I)
NO_EXTRA = re.compile(r"(no|without|negligible|zero) (additional|extra) (computational )?(cost|overhead|NFEs?|"
                      r"network evaluations|function evaluations|inference cost)", re.I)
TOP_VENUE = re.compile(r"NeurIPS|ICML|ICLR|CVPR|ICCV|ECCV|AAAI|IJCAI|SIGGRAPH|TPAMI|ACL|EMNLP|NAACL|Interspeech|"
                       r"ICASSP|TMLR|JMLR|Neural Information Processing|Learning Representations|"
                       r"Computer Vision and Pattern|Machine Learning", re.I)
SURVEY = re.compile(r"\b(survey|review|overview|tutorial)\b", re.I)
GITHUB = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+", re.I)


def expand(goal: str, queries: list[str] | None, max_queries: int = 10) -> list[str]:
    base = [q.strip() for q in (queries or []) if q and q.strip()]
    if not base and goal.isascii():
        base = [goal]
    out = list(base)
    for q in base:
        low = q.lower()
        for group in SYNONYMS:
            for word in group:
                if word in low:
                    out += [re.sub(re.escape(word), alt, q, flags=re.I) for alt in group if alt != word]
    return list(dict.fromkeys(out))[:max_queries]


RESULT = re.compile(r"\b(we|our|ours|achiev\w*|attain\w*|yield\w*|outperform\w*|surpass\w*|improv\w*|reduc\w*|"
                    r"show\w*|demonstrat\w*|experiments?|results?|notably)\b", re.I)
PROBLEM = re.compile(r"^(however|but|although|existing|current|prior|previous|despite)\b", re.I)


def claims(abstract: str, limit: int = 3) -> list[str]:
    """Abstract sentences that state the method's own measured gain, best first (not the problem statement)."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z$])", abstract or "")]
    tiers: list[list[str]] = [[], [], []]
    for s in sents:
        if not GAIN.search(s) or PROBLEM.search(s):
            continue
        num, res = bool(NUMBER.search(s)), bool(RESULT.search(s))
        if num and res:
            tiers[0].append(s)
        elif num or (res and re.search(r"outperform|state-of-the-art|SOTA|surpass", s, re.I)):
            tiers[1 if res else 2].append(s)
    return [h[:300] for h in dict.fromkeys(tiers[0] + tiers[1] + tiers[2])][:limit]


def profile(p: Paper) -> dict[str, Any]:
    text = f"{p.title}. {p.abstract} {p.comment or ''}"
    code = GITHUB.search(text)
    return {
        "training_free": bool(TRAINING_FREE.search(text)) and not NEEDS_TRAINING.search(text),
        "needs_training": bool(NEEDS_TRAINING.search(text)),
        "extra_model_calls": bool(EXTRA_CALLS.search(text)) and not NO_EXTRA.search(text),
        "no_extra_cost_claimed": bool(NO_EXTRA.search(text)),
        "claims": claims(p.abstract),
        "top_venue": bool(TOP_VENUE.search(f"{p.venue or ''} {p.comment or ''}")),
        "code": code.group(0).rstrip(".") if code else None,
        "survey": bool(SURVEY.search(p.title)),
    }


def _baseline_citers(name: str, year_from: int, limit: int = 100) -> list[Paper]:
    """Recent papers citing the baseline method: OpenAlex finds the baseline by name, then its citing works."""
    oa = sources.OpenAlex()
    try:
        hits = oa.search(name, 5)
        if not hits:
            return []
        base = max(hits[:3], key=lambda p: p.citation_count or 0)
        if not base.openalex_id:
            return []
        return oa.works_filter(f"cites:{base.openalex_id},from_publication_date:{year_from}-01-01", limit)
    finally:
        oa.close()


_VEC: dict[str, Any] = {}     # title -> vector, per server process: repeated research does not re-embed
SEMANTIC_MAX = 160            # candidates embedded per call (a static model embeds them in milliseconds)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9-]{3,}", text.lower()))


def _semantic(goal: str, papers: list[Paper]) -> list[float]:
    try:
        import numpy as np

        from kb.process.embed import available, encode_docs, encode_query

        if not available():
            raise RuntimeError("model2vec not installed")
        q = encode_query(goal).astype(np.float32)
        todo = [p for p in papers if p.title not in _VEC]
        if todo:   # title + first sentences carry the method
            vecs = encode_docs([f"{p.title}. {p.abstract[:500]}" for p in todo])
            _VEC.update({p.title: v.astype(np.float32) for p, v in zip(todo, vecs, strict=True)})
        return [float(_VEC[p.title] @ q) for p in papers]
    except Exception as e:   # no embedding model: fall back to word overlap
        log.info("semantic ranking unavailable: %s", e)
        words = {w for w in re.findall(r"\w{4,}", goal.lower())}
        return [len(words & set(re.findall(r"\w{4,}", f"{p.title} {p.abstract}".lower()))) / max(1, len(words))
                for p in papers]


def score(sim: float, prof: dict[str, Any], p: Paper, today: date) -> float:
    s = sim
    s += 0.05 if prof["claims"] else 0.0
    s += 0.04 if prof["training_free"] else 0.0
    s += 0.03 if prof["top_venue"] else 0.0
    s += 0.02 if prof["code"] else 0.0
    s += 0.02 if prof["no_extra_cost_claimed"] else 0.0
    s -= 0.03 if prof["extra_model_calls"] else 0.0
    s -= 0.08 if prof["survey"] else 0.0
    s += 0.01 * math.log10(1 + (p.citation_count or 0))
    if p.published and (today - p.published).days < 365:
        s += 0.02
    return s


def find_methods(db: Database, goal: str, queries: list[str] | None = None, baselines: list[str] | None = None,
                 years: int = 2, limit: int = 12) -> dict[str, Any]:
    from kb.critic.scoring import grade_and_save
    from kb.research.find import _COOLDOWN, COOLDOWN, _key, brief

    qs = expand(goal, queries)
    if not qs:
        return {"error": "pass `queries`: 3-8 short English phrasings of the goal (other words for the same thing: "
                         "solver / sampler, method names, the metric to improve)"}
    today = date.today()
    year_from = today.year - years
    jobs: dict[str, Any] = {}
    pool = ThreadPoolExecutor(max_workers=8)
    for q in qs:
        jobs[f"openalex: {q}"] = pool.submit(lambda q=q: _openalex(q, year_from))
        if _COOLDOWN.get("arxiv", 0) <= time.monotonic():
            jobs[f"arxiv: {q}"] = pool.submit(sources.arxiv_search, q, 40, year_from)
    for b in (baselines or [])[:4]:
        jobs[f"cites {b}"] = pool.submit(_baseline_citers, b, year_from)
    done, _ = wait(jobs.values(), timeout=DEADLINE)
    pool.shutdown(wait=False, cancel_futures=True)

    status: dict[str, str] = {}
    found: dict[str, Paper] = {}
    via: dict[str, set[str]] = {}
    for name, fut in jobs.items():
        if fut not in done:
            status[name] = "timeout"
            continue
        try:
            papers = fut.result()
        except Exception as e:
            status[name] = f"failed: {type(e).__name__}"
            if "429" in str(e) and name.startswith("arxiv"):
                _COOLDOWN["arxiv"] = time.monotonic() + COOLDOWN
            continue
        status[name] = f"ok ({len(papers)})"
        for p in papers:
            k = _key(p)
            if k not in found or len(p.abstract) > len(found[k].abstract):
                found[k] = p
            via.setdefault(k, set()).add("citing baseline" if name.startswith("cites") else "search")
    from kb.textutil import normalize

    by_title: dict[str, tuple[str, Paper]] = {}   # one paper reached as arXiv id, DOI and conference version
    for k, p in found.items():
        t = normalize(p.title)
        if t not in by_title or (p.arxiv_id and not by_title[t][1].arxiv_id) or len(p.abstract) > len(by_title[t][1].abstract) + 200:
            if t in by_title:
                via.setdefault(k, set()).update(via.get(by_title[t][0], set()))
                p.venue = p.venue or by_title[t][1].venue
                p.citation_count = max(p.citation_count or 0, by_title[t][1].citation_count or 0)
            by_title[t] = (k, p)
        else:
            kept = by_title[t][1]
            kept.venue = kept.venue or p.venue
            kept.citation_count = max(kept.citation_count or 0, p.citation_count or 0)
            via.setdefault(by_title[t][0], set()).update(via.get(k, set()))
    cands = list(by_title.values())
    if len(cands) > SEMANTIC_MAX:   # cheap word overlap first, so only plausible ones are embedded
        want = _words(goal + " " + " ".join(qs))
        cands.sort(key=lambda kp: len(want & _words(f"{kp[1].title} {kp[1].abstract}")), reverse=True)
        cands = cands[:SEMANTIC_MAX]
    if not cands:
        return {"goal": goal, "queries": qs, "sources": status, "results": [],
                "next": "nothing found: rephrase (other terms, method family names) or research_sweep(platforms=['web'])"}

    sims = _semantic(goal + " " + " ".join(queries or []), [p for _, p in cands])
    ranked = []
    for (k, p), sim in zip(cands, sims, strict=True):
        prof = profile(p)
        ranked.append((score(sim, prof, p, today), sim, k, p, prof))
    ranked.sort(key=lambda r: r[0], reverse=True)

    out = []
    for total, sim, k, p, prof in ranked[:limit]:
        pid, _ = db.upsert_paper(p)
        grade_and_save(db, pid)
        b = brief(db, pid, abstract_chars=300)
        out.append({
            "id": pid, "title": b["title"], "year": b["year"], "published": str(p.published or ""),
            "venue": b["venue"], "citations": b["citations"], "grade": b["grade"], "url": b["url"],
            "relevance": round(sim, 3), "score": round(total, 3), "found_via": sorted(via.get(k, [])),
            "claimed_gains": prof["claims"], "training_free": prof["training_free"], "needs_training": prof["needs_training"],
            "extra_model_calls": prof["extra_model_calls"], "code": prof["code"], "survey": prof["survey"],
        })
    db.commit()
    return {
        "goal": goal, "queries": qs, "candidates_ranked": len(ranked), "sources": status, "results": out,
        "next": ("These are claims from abstracts. For the top 3-5: fetch(paper_ids=…) and evidence(paper_id=…) to get the "
                 "measured gain with its conditions (model, steps, metric) and the extra cost; github_search the title "
                 "for code; check it applies to the user's setup. Report only methods with a measured gain; say "
                 "plainly when none was shown on the user's model."),
    }


def _openalex(query: str, year_from: int) -> list[Paper]:
    oa = sources.OpenAlex()
    try:
        return oa.search(query, 50, year_from)
    finally:
        oa.close()
