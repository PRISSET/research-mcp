"""Research MCP server (mcp 2.x MCPServer). Run: `uv run research-mcp` (stdio)."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

from mcp.server.mcpserver import MCPServer

from kb import notebook
from kb.store import Database

T = TypeVar("T")

INSTRUCTIONS = """\
Research engine for any field. It searches papers (OpenAlex, arXiv, Semantic Scholar), GitHub (repos, issues,
PRs), Hugging Face (models, datasets, spaces), forums (Hacker News, Stack Overflow) and the web, reads any of
them in full, grades evidence (A official ... D disputed) and keeps findings in one Markdown file, RESEARCH.md.

Rules that matter most:
- Your training data is out of date. Before recommending, choosing, installing or configuring a model, library or
  tool, look it up here — research_sweep / hf_search / github_search, and check_versions for what is already in
  use. Never assume a model or version does not exist because you do not know it.
- Recommending a model: call latest_models for the task first and say when each candidate was released; prefer
  current releases unless an older model is clearly better for the user's constraints (say why).
- Before telling the user which weights to download or load: model_variants with their GPU. Do not default to
  bf16 when it does not fit or barely fits — offer FP8 / INT8 (near-lossless, half the memory), state the quality
  cost and whether the GPU computes FP8 natively or only stores it. 4-bit and below only when 8-bit cannot fit.
- Before promising that a model will run or train on the user's hardware: estimate_resources (VRAM breakdown,
  fits or not, tokens/s, training hours). Choosing between models: compare_models (sizes, context, licence,
  card benchmarks). Before using a dataset: dataset_info. Before reimplementing a paper: paper_code.
- When a command, install, build or run fails: call diagnose_error with the exact error text (and library +
  version) BEFORE trying fixes by guessing. Closed issues and merged PRs usually name the fix and the version.
- If you have your own web search / fetch tools, use them too, and pass the useful pages to fetch_url so they are
  indexed and can be cited in the notebook.

Research workflow:
0. research_brief(topic, …what you know) — if anything important is missing (goal, environment, baseline to beat,
   how "better" is judged, constraints such as speed or memory), ASK the user and repeat until it is ready.
1. notebook_read(query=...) — reuse earlier research; notebook_refresh on a stale section.
2. notebook_start(title, question).
3. research_sweep(query) for the landscape. When the user wants a solution or an improvement, find_methods(goal,
   queries=[several phrasings], baselines=[what to beat]) — it finds the strong recent methods with their claimed
   gains. Then find_papers / github_search / hf_search / forum_search / web_search with focused English queries;
   follow_citations on key papers.
4. Read before concluding: fetch_url (any page, repo, issue, model card), fetch_fulltext (papers), then
   search_fulltext(question, section=...), read_passage, find_paper_numbers. Never take numbers from snippets.
5. critique / what_others_say / verify_claim / check_versions before trusting a strong claim or an old setup.
6. notebook_add_source for every source worth keeping (what it says, numbers with conditions, limitations,
   applicability), notebook_write(summary=...) with conclusions citing source ids.
Write notes in the user's language. Only sources returned by these tools can be added — never invent citations.

Answer like an expert, not a search engine: lead with the few strongest options, each with its measured gain
(number, metric, model and steps it was measured on, source), its cost (extra model calls, speed, memory), whether
it needs training, code availability and how to plug it into the user's pipeline. No generic advice and no long
lists of weak candidates; if nothing measured beats the baseline on the user's setup, say so in one line and give
the experiment that would decide it.
"""

def _instructions() -> str:
    from kb.update import update_note

    return INSTRUCTIONS + update_note()


mcp = MCPServer("research", instructions=_instructions())


def _with_db(fn: Callable[[Database], T]) -> T:
    db = Database()
    try:
        return fn(db)
    finally:
        db.close()


def _section_papers(section: str | None) -> set[str] | None:
    return set(notebook.read_section(section)["papers"]) if section else None


def _safe(fn: Callable[[], T]) -> T | dict[str, str]:
    try:
        return fn()
    except notebook.NotebookError as e:
        return {"error": str(e)}


# ---------------- finding ----------------

@mcp.tool()
def research_sweep(query: str, platforms: list[str] | None = None, per_platform: int = 6) -> dict:
    """Ask every platform at once — start here when planning, choosing a model / library / approach, or entering
    an unfamiliar topic: top papers, Hugging Face models, GitHub repos and issues, web pages and forum threads,
    each graded. Then read the promising ones with fetch_url / fetch_fulltext.

    Args:
        query: short English query, e.g. "talking head lip sync", "MiniMax H3 LoRA training"
        platforms: subset of ["papers", "models", "repos", "issues", "web", "forums"] (default: all)
        per_platform: results per platform (3-15)
    """
    from kb.research.sweep import research_sweep as _sweep

    return _sweep(query, platforms, per_platform)


@mcp.tool()
def web_search(query: str, limit: int = 10) -> dict:
    """General web search (Brave / Tavily if a key is set, otherwise DuckDuckGo): docs, blogs, vendor pages,
    announcements. Read results with fetch_url. If you have your own web search tool, use it as well."""
    from kb.research.web import web_search as _web

    return _with_db(lambda db: _web(db, query, min(limit, 20)))


@mcp.tool()
def github_search(query: str, kind: str = "repos", repo: str | None = None, state: str | None = None,
                  limit: int = 10) -> dict:
    """Search GitHub. kind="repos": projects with stars, last push, license, archived flag. kind="issues" / "prs":
    real-world problems, workarounds and fixes (put an error message in quotes for an exact match).

    Args:
        repo: limit issues/PRs to "owner/name";  state: "open" | "closed"
    """
    from kb.research.platforms import github_search as _gh

    kind = kind if kind in ("repos", "issues", "prs") else "repos"
    return _with_db(lambda db: _gh(db, query, kind, repo, min(limit, 30), state))


@mcp.tool()
def hf_search(query: str, kind: str = "models", sort: str = "trending", limit: int = 10) -> dict:
    """Search Hugging Face: models / datasets / spaces / papers, with release date and age, downloads, likes,
    license, task, base model and relation (quantized / finetune / adapter). Matches repo names, so use short
    name-like queries ("MiniMax-H3", "whisper russian", "lipsync"). For "what is newest for task X" use
    latest_models instead.

    Args:
        kind: models | datasets | spaces | papers;  sort: trending | downloads | likes | created | recent
    """
    from kb.research.platforms import hf_search as _hf

    kind = kind if kind in ("models", "datasets", "spaces", "papers") else "models"
    return _with_db(lambda db: _hf(db, query, kind, sort, min(limit, 30)))


@mcp.tool()
def forum_search(query: str, sites: list[str] | None = None, limit: int = 8) -> dict:
    """Community experience: Hacker News discussions and Stack Overflow questions (score, accepted answer).

    Args:
        sites: subset of ["hackernews", "stackoverflow"] (default: both)
    """
    from kb.research.platforms import forum_search as _forum

    return _with_db(lambda db: _forum(db, query, sites, min(limit, 20)))


@mcp.tool()
def find_papers(query: str, limit: int = 15, year_from: int | None = None, sources: list[str] | None = None) -> dict:
    """Search papers live in OpenAlex (all fields), arXiv and Semantic Scholar plus the local cache; duplicates
    merged, ranked by agreement between sources and citations, each graded A-D. Results are stored locally.

    Args:
        query: focused English query, e.g. "speculative decoding acceptance rate" (run several narrow queries
            rather than one broad one)
        limit: results to return (<= 40)
        year_from: only papers from this year on
        sources: subset of ["openalex", "arxiv", "s2", "local"] (default: all)
    """
    from kb.research.find import SOURCES
    from kb.research.find import find_papers as _find

    use = tuple(s for s in (sources or SOURCES) if s in SOURCES) or SOURCES
    return _with_db(lambda db: _find(db, query, min(limit, 40), year_from, use, on_paper=notebook.sections_with))


@mcp.tool()
def find_methods(goal: str, queries: list[str] | None = None, baselines: list[str] | None = None, years: int = 2,
                 limit: int = 12, section: str | None = None, save_top: int = 6) -> dict:
    """The strongest recent METHODS for a concrete goal ("a better sampler than X at the same cost", "reduce
    hallucinations in Y without retraining"). Searches many phrasings plus the papers that cite the baseline you
    want to beat, ranks by meaning against the goal and by evidence, and returns for each: the claimed gain
    (sentences with numbers from the abstract), training-free or not, extra model calls, venue, code link.
    Use it instead of find_papers when the user wants solutions, not background.

    Args:
        goal: the goal with its constraints, one or two English sentences (what to improve, on what, at what cost)
        queries: 3-8 short English phrasings using different words for the same thing (solver / sampler /
            integrator, method-family names, the metric) — the best paper often uses a word you did not
        baselines: the method(s) to beat, by name ("DPM-Solver++", "UniPC") — newer work cites and compares to them
        years: how far back to look (default 2)
        section: a RESEARCH.md topic (from notebook_start): the top `save_top` candidates are written there right
            away with their claimed gains, marked as not yet verified — refine them with notebook_add_source(replace=true)
    """
    from kb.research.methods import find_methods as _find

    def run(db: Database) -> dict:
        out = _find(db, goal, queries, baselines, max(1, min(years, 6)), min(limit, 25))
        if section and out.get("results"):
            out["saved_to_notebook"] = _save_candidates(db, section, out["results"][:max(0, min(save_top, 12))])
        return out

    return _with_db(run)


def _save_candidates(db: Database, section: str, results: list[dict]) -> list[str] | dict:
    ru = os.environ.get("RESEARCH_LANG", "en").lower().startswith("ru")
    yes, no = ("да", "нет") if ru else ("yes", "no")
    saved = []
    for r in results:
        p = db.get_paper(r["id"])
        if p is None:
            continue
        gains = " / ".join(r["claimed_gains"]) or ("в аннотации цифр нет" if ru else "no numbers in the abstract")
        lines = [
            f"- **{'Заявленный прирост (из аннотации)' if ru else 'Claimed gain (abstract)'}:** {gains}",
            f"- **{'Без дообучения' if ru else 'Training-free'}:** {yes if r['training_free'] else no}"
            + (f" ({'нужно обучить вспомогательную часть' if ru else 'needs some training'})" if r["needs_training"] else "")
            + (f" · **{'лишние вызовы модели' if ru else 'extra model calls'}:** {yes}" if r["extra_model_calls"] else ""),
            f"- **{'Код' if ru else 'Code'}:** {r['code'] or ('не указан в аннотации' if ru else 'not in the abstract')}",
            f"- **{'Статус' if ru else 'Status'}:** "
            + ("кандидат — цифры и применимость ещё не проверены по полному тексту" if ru
               else "candidate — numbers and applicability not yet checked in the full text"),
        ]
        res = _safe(lambda p=p, r=r, lines=lines: notebook.add_paper(section, {**p, "grade": r["grade"]}, "\n".join(lines)))
        if isinstance(res, dict) and "error" in res:
            return res
        saved.append(r["id"])
    return saved


@mcp.tool()
def research_brief(topic: str, goal: str | None = None, environment: str | None = None, baseline: str | None = None,
                   success_metric: str | None = None, constraints: str | None = None, deliverable: str | None = None,
                   already_tried: str | None = None, user_declined: list[str] | None = None) -> dict:
    """Call FIRST for any research, planning or "find the best way to…" request. Fill every field you can from
    the conversation; it returns what is still missing as concrete questions for the user. Ask them, call again
    with the answers, and search only when `ready` is true — a precise brief is what makes the answer strong.

    Args:
        topic: the user's request, verbatim
        goal / environment / baseline / success_metric / constraints / deliverable / already_tried: what you know
            (leave empty when unknown — do not guess)
        user_declined: fields the user could not or did not want to answer
    """
    from kb.research.brief import research_brief as _brief

    return _brief(topic, goal, environment, baseline, success_metric, constraints, deliverable, already_tried,
                  user_declined)


@mcp.tool()
def follow_citations(paper_id: str, direction: str = "references", limit: int = 20) -> dict:
    """Walk the citation graph: "references" = what the paper builds on (foundations, baselines), "citations" =
    later work that cites it (follow-ups, reproductions, critiques). Influential and most-cited first.

    Args:
        paper_id: id from find_papers, e.g. "arxiv:2211.17192" or "doi:10.1038/..."
    """
    from kb.research.graph import follow_citations as _follow

    return _with_db(lambda db: _follow(db, paper_id, direction, min(limit, 50)))


@mcp.tool()
def get_source(source_id: str) -> dict:
    """Everything stored about one source (paper, repo, issue, model, page): metadata, platform data (stars,
    downloads, license, state), evidence grade with flags, full-text status, notebook sections citing it."""
    def run(db: Database) -> dict:
        p = db.get_paper(source_id)
        if p is None:
            return {"error": f"unknown source {source_id}: find it with the search tools first"}
        from kb.research.find import brief

        return {**p, **brief(db, p["id"], abstract_chars=5000), "in_notebook": notebook.sections_with(p["id"])}

    return _with_db(run)


# ---------------- reading ----------------

@mcp.tool()
def fetch_url(url: str | None = None, urls: list[str] | None = None) -> dict:
    """Read any page and index it for search_fulltext: docs, blogs, vendor sites, PDFs, GitHub repos (README +
    release notes), files and issue / PR threads (all comments, maintainers marked), Hugging Face model cards
    (with license, files, parameters), arXiv papers, Hacker News threads. Returns the text (first 12k chars),
    an outline and the source id for notebook_add_source. Pass `urls` (up to 12) to read several pages in
    parallel (5k chars each)."""
    from kb.research.web import fetch_many
    from kb.research.web import fetch_url as _fetch

    if urls:
        return {"results": fetch_many(urls + ([url] if url else []))}
    if not url:
        return {"error": "pass url or urls"}

    def run(db: Database) -> dict:
        try:
            return _fetch(db, url)
        except Exception as e:
            return {"url": url, "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "next": "try your own fetch tool, or another page with the same information"}

    return _with_db(run)


@mcp.tool()
def fetch_fulltext(paper_ids: list[str], refresh: bool = False) -> dict:
    """Download and index full texts (arXiv HTML/PDF or an open-access PDF): split into passages, extract numbers,
    embed for semantic search. Needed before search_fulltext / find_paper_numbers can see a paper. Up to 10 ids
    per call; papers without an open full text are reported (then work from the abstract)."""
    from kb.process.fulltext import fetch_papers

    return _with_db(lambda db: fetch_papers(db, paper_ids[:10], refresh=refresh))


@mcp.tool()
def search_fulltext(
    query: str,
    k: int = 8,
    section: str | None = None,
    paper_ids: list[str] | None = None,
    year_from: int | None = None,
    grades: str | None = None,
    kinds: list[str] | None = None,
) -> list[dict] | dict:
    """Semantic + keyword search INSIDE fetched full texts (methods, tables, ablations, limitations), reranked.
    Works across languages. Every hit has a citation (paper, section) and the paper's grade.

    Args:
        query: a natural-language question, e.g. "how does acceptance rate change with draft model size"
        section: limit to papers of this notebook section
        paper_ids: limit to these papers
        grades: e.g. "AB" to keep only strong evidence
        kinds: subset of ["text", "table", "figure"]; tables carry the actual numbers
    """
    from kb.process.search import search

    def run(db: Database) -> list[dict] | dict:
        try:
            allowed = _section_papers(section)
        except notebook.NotebookError as e:
            return {"error": str(e)}
        if paper_ids:
            ids = {(db.get_paper(p) or {"id": p})["id"] for p in paper_ids}
            allowed = ids if allowed is None else allowed & ids
        return search(db, query, k=min(k, 25), paper_ids=allowed, year_from=year_from, grades=grades,
                      kinds=tuple(kinds or ("text", "table", "figure")))

    return _with_db(run)


@mcp.tool()
def read_passage(chunk_id: int, window: int = 1) -> list[dict]:
    """A search_fulltext hit with its neighbouring passages, to read a claim in context before citing it."""
    from kb.process.search import context

    return _with_db(lambda db: context(db, chunk_id, min(window, 4)))


@mcp.tool()
def find_paper_numbers(
    kind: str | None = None,
    hardware: str | None = None,
    model: str | None = None,
    text: str | None = None,
    paper_id: str | None = None,
    limit: int = 30,
) -> list[dict]:
    """Numbers extracted from fetched full texts, verbatim with their conditions: speedups ("2.3x on H100"),
    memory, step counts, latency, benchmark scores (accuracy, F1, BLEU, WER, FID, ...). Use to compare results
    across papers and check whether they were measured under comparable conditions.

    Args:
        kind: speedup | memory | steps | latency | quality_metric
        hardware: e.g. "H100";  model: e.g. "Llama";  text: words that must appear;  paper_id: one paper
    """
    from kb.process.claims import find

    def run(db: Database) -> list[dict]:
        pid = (db.get_paper(paper_id) or {"id": paper_id})["id"] if paper_id else None
        return find(db, kind, hardware, model, text, pid, min(limit, 100))

    return _with_db(run)


# ---------------- errors & versions ----------------

@mcp.tool()
def diagnose_error(error: str, library: str | None = None, version: str | None = None,
                   context: str | None = None) -> dict:
    """Call this FIRST when something fails, before guessing at fixes. Searches the exact error message in GitHub
    issues and PRs, Stack Overflow and the web at once (shortening the phrase until something matches), checks
    whether the library version is outdated, and says whether a fix exists (merged PR / closed issue).

    Args:
        error: the error text or traceback as printed
        library: the package / project involved, e.g. "vllm";  version: the installed version, e.g. "0.29.0"
        context: what was being done (command, model, hardware) — echoed back for the record
    """
    from kb.research.versions import diagnose_error as _diag

    return _diag(Database, error, library, version, context)


@mcp.tool()
def check_versions(items: list[str], focus: str | None = None) -> dict:
    """Are these current? PyPI packages ("vllm==0.8.1", "torch"), Hugging Face models
    ("Qwen/Qwen3-Omni-30B-A3B-Instruct": newer models of the same family and newer alternatives for the same
    task) and GitHub repos ("gh:vllm-project/vllm@v0.9.0"). For outdated pinned versions it digests the release
    notes in between: breaking changes, and lines about `focus` (e.g. "Qwen3-Omni audio").
    Use before choosing a model, when setting up, and when debugging."""
    from kb.research.versions import check_versions as _check

    return _check(items, focus)


# ---------------- choosing models ----------------

@mcp.tool()
def latest_models(task: str, months: int = 9, limit: int = 10) -> dict:
    """What is new for a task: the newest strong base models (labs' own releases first), the hottest
    derivatives (fine-tunes, quantizations, LoRAs) and the older popular ones for comparison, with release
    dates, plus leaderboards to check. Call before recommending any model.

    Args:
        task: free text or a Hugging Face task, e.g. "text-to-video", "speech recognition", "LLM", "TTS"
        months: how recent counts as new
    """
    from kb.research.models import latest_models as _latest

    return _latest(task, max(1, min(months, 36)), min(limit, 25))


@mcp.tool()
def model_variants(model_id: str, vram_gb: float | None = None, gpu: str | None = None) -> dict:
    """Which weights to download: every precision in the model's repo (bf16 / fp8 / int8 / w6a8 / GGUF …) and
    the community's quantized versions, with sizes, whether each fits the GPU (detected automatically, or pass
    vram_gb / gpu for another card) and the quality cost. Recommends the best quality that fits, offers FP8 /
    INT8 when bf16 is tight, and says whether the GPU computes FP8 / FP4 natively.

    Args:
        model_id: Hugging Face repo, e.g. "Qwen/Qwen3-8B" or "Comfy-Org/MiniMax-H3"
        vram_gb, gpu: plan for another GPU, e.g. vram_gb=24, gpu="RTX 4090"
    """
    from kb.research.models import model_variants as _variants

    try:
        return _variants(model_id, vram_gb, gpu)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


# ---------------- critique ----------------

@mcp.tool()
def estimate_resources(model_id: str, mode: str = "inference", precision: str | None = None, seq_len: int = 4096,
                       batch: int = 1, lora_rank: int = 16, gradient_checkpointing: bool = True,
                       train_tokens: float | None = None, gpu: str | None = None, vram_gb: float | None = None,
                       num_gpus: int = 1, kv_cache: str = "bf16") -> dict:
    """Memory and time for running or training a Hugging Face model, from its real config and parameter count:
    VRAM breakdown (weights, KV cache, activations, logits, gradients, optimizer states), whether it fits the GPU
    (detected, or pass gpu / vram_gb), decode tokens/s, training hours for train_tokens, and what to change if it
    does not fit. Call before promising that something will run or train on the user's hardware.

    Args:
        model_id: "Qwen/Qwen3-8B" or a Hugging Face URL
        mode: inference | lora | qlora | full
        precision: bf16 | fp16 | fp8 | int8 | q8 | q6 | q5 | q4 | nf4 | awq | gptq | fp4 (default bf16; nf4 for qlora)
        seq_len, batch: context per sequence and sequences per step
        train_tokens: total training tokens (dataset tokens x epochs) for a time estimate
        num_gpus: tensor parallel (inference) or FSDP / ZeRO-3 (full) across this many GPUs
        kv_cache: bf16 | fp8
    """
    from kb.research.ml import estimate_resources as _est

    return _est(model_id, mode, precision, max(1, seq_len), max(1, batch), max(1, lora_rank), gradient_checkpointing,
                train_tokens, gpu, vram_gb, max(1, num_gpus), kv_cache)


@mcp.tool()
def compare_models(model_ids: list[str]) -> dict:
    """Two to six Hugging Face models side by side: release date, parameters (active for MoE), architecture,
    context length, license and gating, downloads — and the benchmark tables from each model card (gated cards
    included). Card numbers are self-reported: compare only matching settings."""
    from kb.research.ml import compare_models as _cmp

    return _cmp(model_ids)


@mcp.tool()
def dataset_info(dataset_id: str, sample_rows: int = 3) -> dict:
    """A Hugging Face dataset before using it: configs and splits with row counts, size, columns and types, a few
    sample rows, license, gating, and the most used models trained on it."""
    from kb.research.ml import dataset_info as _ds

    return _ds(dataset_id, max(0, min(sample_rows, 10)))


@mcp.tool()
def paper_code(paper: str) -> dict:
    """Code and artifacts for an arXiv paper: the official repository (with stars), project page, and the Hugging
    Face models, datasets and Spaces built on it; otherwise GitHub repos that cite it, the likely official one first.

    Args:
        paper: arXiv id or URL ("2509.00036", "https://arxiv.org/abs/2509.00036", "arxiv:2509.00036")
    """
    from kb.research.ml import paper_code as _code

    return _code(paper)


@mcp.tool()
def critique(source_id: str) -> dict:
    """Evidence grade A-D with reasons and flags. Papers: peer review, venue, code, citations, OpenReview status
    (checked now), vendor self-report. Other sources: official org or not, adoption, maintenance (stale, archived),
    license, unsafe pickle weights, open vs fixed issues."""
    from kb.critic.openreview import check_openreview
    from kb.critic.scoring import grade_and_save

    def run(db: Database) -> dict:
        p = db.get_paper(source_id)
        if p is None:
            return {"error": f"unknown source {source_id}: find it with the search tools first"}
        if p.get("kind", "paper") != "paper":
            import json

            from kb.critic.sources import grade_doc

            g = grade_doc(p["kind"], p.get("extra") or {}, p["id"])
            db.conn.execute("INSERT OR REPLACE INTO critique (source_id, kind, grade, reasons, flags) VALUES (?, ?, ?, ?, ?)",
                            (p["id"], p["kind"], g["grade"], json.dumps(g["reasons"]), json.dumps(g["flags"])))
            db.commit()
            return {"source": p["title"], "id": p["id"], "kind": p["kind"], **g}
        try:
            check_openreview(db, paper_ids=[p["id"]])
        except Exception:
            pass
        r = grade_and_save(db, p["id"])
        db.commit()
        return {"paper": p["title"], "id": p["id"], **(r or {})}

    return _with_db(run)


@mcp.tool()
def verify_claim(claim: str, section: str | None = None) -> dict:
    """Do not take claims on faith: related papers with grades, full-text passages and extracted numbers about a
    statement, with a verdict on the strength of the evidence. E.g. "4-bit quantization loses <1% on MMLU".

    Args:
        section: look only at papers of this notebook section
    """
    from kb.critic.claims import verify_claim as _verify

    def run(db: Database) -> dict:
        try:
            return _verify(db, claim, _section_papers(section))
        except notebook.NotebookError as e:
            return {"error": str(e)}

    return _with_db(run)


@mcp.tool()
def what_others_say(paper_id: str, stance: str | None = None, limit: int = 20) -> dict:
    """What later papers say about a paper (Semantic Scholar citation contexts): critical, possible_critical,
    comparative, neutral. Independent reports outweigh the authors' own claims.

    Args:
        stance: critical | possible_critical | comparative | neutral (default: all, critical first)
    """
    from kb.critic.citation_contexts import what_others_say as _what

    return _with_db(lambda db: _what(db, paper_id, stance, min(limit, 100)))


# ---------------- RESEARCH.md ----------------

@mcp.tool()
def notebook_read(section: str | None = None, query: str | None = None) -> dict | list[dict]:
    """Read RESEARCH.md. No arguments: list of topics (question, paper count, last update) and the base size.
    section: the full Markdown of one topic. query: where earlier research mentions these words."""
    if query:
        return notebook.search(query)
    if section:
        return _safe(lambda: notebook.read_section(section))
    from kb.research.refresh import stale_sections

    out = {**notebook.overview(), "local_base": _with_db(lambda db: db.stats())}
    stale = stale_sections()
    if stale:
        out["stale"] = stale
        out["hint"] = "Sections older than 30 days: notebook_refresh(section) before relying on them."
    return out


@mcp.tool()
def notebook_refresh(section: str, since: str | None = None) -> dict:
    """What appeared since a topic was last updated: re-runs its saved queries for papers, Hugging Face models,
    GitHub repos and issues published after its date, and checks the models / repos it cites for newer
    versions. Use before relying on an old section.

    Args:
        since: ISO date to look from (default: the section's last update)
    """
    from kb.research.refresh import refresh

    return _safe(lambda: refresh(section, since))


@mcp.tool()
def notebook_start(title: str, question: str, queries: list[str] | None = None) -> dict:
    """Open a topic in RESEARCH.md (or return the existing one with the same title, to build on it).

    Args:
        title: short topic name in the user's language, e.g. "Ускорение MiniMax H3"
        question: what the research must answer
        queries: search queries used so far (shown in the section header)
    """
    return _safe(lambda: notebook.start(title, question, queries))


@mcp.tool()
def notebook_add_source(section: str, source_id: str, notes: str, replace: bool = False) -> dict:
    """Add a source (paper, repo, issue, model, page, thread) to a topic. Its title line (link, type, year, grade,
    stars / downloads / citations, license, state) is generated from the base; `notes` is your Markdown analysis:
    "- **Суть:** …\\n- **Цифры:** 2.3x on A100, batch 1 (табл. 3)\\n- **Ограничения:** …\\n- **Применимость:** …"
    A source already in the section is not duplicated; replace=true rewrites its notes."""
    from kb.critic.scoring import grade_and_save

    def run(db: Database) -> dict:
        p = db.get_paper(source_id)
        if p is None:
            return {"error": f"unknown source {source_id}: only sources found through these tools can be added"}
        row = db.conn.execute("SELECT grade FROM critique WHERE source_id = ?", (p["id"],)).fetchone()
        if row:
            grade = row["grade"]
        elif p.get("kind", "paper") == "paper":
            grade = (grade_and_save(db, p["id"]) or {}).get("grade")
        else:
            from kb.critic.sources import grade_doc

            grade = grade_doc(p["kind"], p.get("extra") or {}, p["id"])["grade"]
        db.commit()
        return _safe(lambda: notebook.add_paper(section, {**p, "grade": grade}, notes, replace))

    return _with_db(run)


@mcp.tool()
def notebook_write(
    section: str,
    summary: str | None = None,
    append: bool = False,
    question: str | None = None,
    add_queries: list[str] | None = None,
    remove_papers: list[str] | None = None,
) -> dict:
    """Update a topic: its conclusions (summary, Markdown; append=true adds to the existing text), the question,
    the list of search queries, or remove papers that turned out irrelevant."""
    return _safe(lambda: notebook.write(section, summary, append, question, add_queries, remove_papers))


# ---------------- prompt ----------------

@mcp.prompt(title="Research a topic")
def research(topic: str, depth: str = "normal") -> str:
    """Research a topic across papers, GitHub, Hugging Face, forums and the web; record it in RESEARCH.md.

    Args:
        topic: what to research or decide, in any language
        depth: quick (about 5 sources) | normal (about 10) | deep (15+, citation chasing and critique of every claim)
    """
    n = {"quick": 5, "deep": 15}.get(depth, 10)
    return (
        f"Research this with the research MCP tools and record everything in RESEARCH.md: {topic}\n\n"
        f"Depth: {depth} — aim for about {n} well-chosen sources in the notebook, of several kinds.\n"
        "Do not answer from memory: your training data is stale; every model, library and number must come "
        "from a source you read now.\n"
        "0. research_brief(topic, …everything you already know). While it is not ready, ask the user its "
        "questions (all at once) and call it again with the answers — never fill gaps by guessing.\n"
        "1. notebook_read(query=…) to reuse earlier research; notebook_start with a short title and the precise "
        "question (include the brief).\n"
        "2. research_sweep for the landscape. If the user wants a solution or an improvement over something, "
        "find_methods(goal, queries=[3-8 phrasings with different words], baselines=[what to beat]). Then focused "
        "find_papers / github_search (repos and issues) / hf_search / forum_search / web_search. Use your own web "
        "search too and fetch_url the useful pages."
        + (" follow_citations on the central papers in both directions." if depth == "deep" else "") + "\n"
        "3. Read before concluding: fetch_url for repos, model cards, docs, issue threads; fetch_fulltext for "
        "papers; search_fulltext(question, section=…) and find_paper_numbers for exact numbers and conditions.\n"
        "4. For model choices: latest_models for the task (state release dates) and model_variants for the user's "
        "GPU (FP8 / INT8 before 4-bit when bf16 does not fit); check_versions on every library you will recommend; "
        + ("critique and what_others_say for every key claim; " if depth == "deep" else
           "critique the sources your conclusions rely on; ")
        + "say when evidence is only C/D.\n"
        "5. notebook_add_source per source (what it says, numbers with conditions and where they come from, "
        "limitations, applicability), then notebook_write(summary=…) with conclusions and a concrete "
        "recommendation, citing source ids.\n"
        "Write the notes in the language of the request. Finish with a short answer and the path of RESEARCH.md: "
        "the 2-5 strongest options first, each with its measured gain (number, metric, conditions, source), cost "
        "(extra model calls / speed / memory), training needed or not, code, and how to integrate it. No filler, "
        "no long lists of weak candidates; if nothing measured beats the baseline, say so and propose the "
        "experiment that decides it."
    )


@mcp.prompt(title="Fix an error")
def fix_error(error: str, library: str = "", version: str = "") -> str:
    """Look an error up before guessing: issues, PRs, Stack Overflow, web, version check.

    Args:
        error: the error text or traceback
        library: package / project involved (optional);  version: installed version (optional)
    """
    return (
        "An error occurred. Do not guess at fixes yet.\n"
        f"1. diagnose_error(error=…, library={library or 'the package involved'!r}, version={version or 'installed'!r}).\n"
        "2. Read the best matching issue / PR in full (fetch_url or the github link), including maintainers' "
        "comments; note the version that contains the fix and any documented workaround.\n"
        "3. check_versions for the libraries and models involved; if outdated, read the release notes first.\n"
        "4. If nothing matches, use your own web search with the exact message, then fetch_url the results.\n"
        "5. Apply the documented fix, verify it, and if it took effort to find, record it in RESEARCH.md "
        "(notebook_start / notebook_add_source / notebook_write).\n\n"
        f"Error:\n{error}"
    )


def main() -> None:
    import logging

    from kb.update import update_and_restart

    update_and_restart()   # new version published? fast-forward and re-exec before anything starts

    # Agents show the server's stderr: keep it to warnings, not one line per HTTP request.
    logging.basicConfig(level=logging.WARNING)
    for name in ("httpx", "httpcore", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:   # background helper: the user's own programs get the CPU first
        os.nice(int(os.environ.get("KB_NICE", "10")))
    except (AttributeError, OSError, ValueError):
        pass
    if os.environ.get("KB_WARMUP", "1") != "0":
        import threading

        from kb.process.search import warm_up

        threading.Thread(target=warm_up, name="warm-up", daemon=True).start()
    mcp.run()


if __name__ == "__main__":
    main()
