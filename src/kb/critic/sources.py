"""Evidence grades for non-paper sources, by what kind of source it is and how alive / adopted it is.

A  official: the lab's own model card, repo or docs (its organisation owns the repo)
B  widely adopted and maintained (many stars / downloads, recent activity), merged fixes
C  community: issues, forum threads, blog posts, small repos — practical but unverified
D  abandoned, archived or contradicted
The agent should still say who wrote a statement: an official card is A for facts about that model, not for
claims about competitors.
"""

from __future__ import annotations

from datetime import date
from typing import Any

# Organisations whose own repos / model cards count as official sources for their models.
OFFICIAL_ORGS = {
    "minimaxai", "minimax-ai", "qwen", "qwenlm", "meta-llama", "facebook", "facebookresearch", "google", "google-deepmind",
    "google-research", "openai", "mistralai", "deepseek-ai", "microsoft", "nvidia", "stabilityai", "stability-ai",
    "black-forest-labs", "wan-ai", "wan-video", "tencent", "tencent-hunyuan", "thudm", "zai-org", "ibm-granite",
    "allenai", "huggingface", "apple", "amazon", "anthropics", "cohereforai", "coherelabs", "01-ai", "baidu",
    "bytedance", "bytedance-seed", "moonshotai", "lightricks", "genmo", "alibaba-pai", "xai-org", "openbmb",
    "internlm", "opengvlab", "salesforce", "vllm-project", "sgl-project", "pytorch", "comfyanonymous", "comfy-org",
    "ggerganov", "ggml-org", "unslothai", "lmstudio-ai", "ollama", "pytorch-labs", "hpcai-tech", "modelscope",
}
OFFICIAL_DOMAINS = ("docs.", "readthedocs.io", "pytorch.org", "huggingface.co/docs", "nvidia.com", "openai.com",
                    "anthropic.com", "ai.google.dev", "minimax.io", "qwenlm.github.io", "mistral.ai", "deepseek.com")


def _age_days(d: str | None) -> int | None:
    if not d:
        return None
    try:
        return (date.today() - date.fromisoformat(d[:10])).days
    except ValueError:
        return None


def grade_doc(kind: str, extra: dict[str, Any], doc_id: str = "") -> dict[str, Any]:
    reasons: list[str] = []
    flags: list[str] = []
    owner = str(extra.get("owner") or "").lower()
    official = owner in OFFICIAL_ORGS
    grade = "C"

    if kind == "repo":
        stars, pushed = extra.get("stars") or 0, _age_days(extra.get("pushed"))
        reasons.append(f"{stars} stars" + (f", last push {extra.get('pushed')}" if extra.get("pushed") else ""))
        if extra.get("archived"):
            grade = "D"
            flags.append("archived: no longer maintained")
        elif official:
            grade = "A"
            reasons.append("official organisation repo")
        elif stars >= 1000 and (pushed is None or pushed < 365):
            grade = "B"
        if pushed is not None and pushed > 365 and not extra.get("archived"):
            flags.append(f"stale: no commits for {pushed // 30} months — check for a maintained fork or successor")
        if extra.get("fork"):
            flags.append("fork: compare with upstream")
        if not extra.get("license") or extra.get("license") == "NOASSERTION":
            flags.append("no clear license")
    elif kind in ("issue", "pr"):
        reasons.append(f"{extra.get('state')} · {extra.get('comments') or 0} comments")
        if kind == "pr" and extra.get("merged"):
            grade = "B"
            reasons.append("merged: the fix is in the codebase (check which release ships it)")
        if extra.get("maintainer_replied"):
            reasons.append("a maintainer replied")
        if kind == "issue" and extra.get("state") == "open":
            flags.append("still open: the problem may be unresolved")
    elif kind in ("model", "dataset", "space"):
        dl, likes = extra.get("downloads") or 0, extra.get("likes") or 0
        reasons.append(f"{dl:,} downloads, {likes} likes")
        if official:
            grade = "A"
            reasons.append("official model card of the authoring organisation")
        elif dl >= 100_000 or likes >= 200:
            grade = "B"
            reasons.append("broad adoption")
        if extra.get("pickle_files") and not extra.get("safetensors"):
            flags.append(f"pickle-only weights can execute code on load: {extra['pickle_files'][:3]}")
        if not extra.get("license"):
            flags.append("no license in the card")
        if extra.get("gated"):
            flags.append("gated: needs accepting terms / a token to download")
        born = _age_days(extra.get("created"))
        age = _age_days(extra.get("updated"))
        if born is not None and born > 365 and kind == "model":
            flags.append(f"released {born // 30} months ago — check latest_models for a newer generation")
        elif age is not None and age > 365:
            flags.append(f"not updated for {age // 30} months — check for a newer version")
    elif kind == "web":
        domain = str(extra.get("domain") or doc_id).lower()
        if any(d in domain for d in OFFICIAL_DOMAINS) or (doc_id.startswith("gh:") and official):
            grade = "B"
            reasons.append("documentation / vendor site: authoritative for its own product")
        else:
            flags.append("web page: check who wrote it and when")
    elif kind == "so":
        reasons.append(f"score {extra.get('score')}, {'answered' if extra.get('answered') else 'no accepted answer'}")
        if not extra.get("answered"):
            flags.append("no accepted answer")
    elif kind == "hn":
        reasons.append(f"{extra.get('points')} points, {extra.get('comments')} comments: opinions, not evidence")
    return {"grade": grade, "reasons": reasons, "flags": flags}
