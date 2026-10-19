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
Research engine for ML and any other field: papers (OpenAlex, arXiv, Semantic Scholar), GitHub, Hugging Face,
Civitai, ComfyUI nodes, YouTube, forums and the web, read in full, graded by evidence (A official ... D disputed), with findings kept in one
Markdown file, RESEARCH.md. What it reads is cached locally and searchable (search_fulltext).

Rules that matter most:
- Your training data is out of date. Before recommending, choosing, installing or configuring a model, library or
  tool, look it up here (research_sweep / hf_search / github_search; check_versions for what is already in use).
  Never assume a model or version does not exist because you do not know it.
- Recommending a model: latest_models for the task first, and say when each candidate was released.
- Which weights to download: model_variants with the user's GPU. Do not default to bf16 when it does not fit or
  barely fits — offer FP8 / INT8 (near-lossless, half the memory) and state the quality cost. 4-bit and below
  only when 8-bit cannot fit. Will it run or train on their hardware: estimate_resources.
- A command, install, build or run failed: diagnose_error with the exact error text BEFORE guessing at fixes.
- Your own web search is the main web search (Reddit, blogs, news): the built-in one is only good with a Tavily /
  Brave key. Pass useful pages to fetch so they are cached and can be cited in the notebook. When a result says
  an optional key would help, tell the user once (`uv run kb keys`).

Research workflow: research_brief (ask the user what it reports missing) -> notebook_read(query=…) to reuse
earlier work -> notebook_write(title=…, question=…) to open a topic -> research_sweep / find_methods / find_papers
-> fetch the promising sources and read them (search_fulltext with English queries, evidence for numbers) ->
source_info(critique=true) on what conclusions rely on -> notebook_add_source per source -> notebook_write(summary=…).
Write notes in the user's language. Only sources returned by these tools can be added — never invent citations.

Be the specialist, not a search box. When the user does practical work in a field (designing an annotation
schema or dataset, training, evaluation, a pipeline), first learn the field: study_topic(topic, terms=[its key
concepts], section=…) searches it facet by facet (foundations, reference datasets / systems, each concept, the
decisions every such project makes, practices, pitfalls, tools) and reads the best sources; write what you learn
with notebook_write(section, knowledge=[…]) and repeat until it reports expert_ready. Then advise proactively:
explain the concepts that matter for their work and what they change, raise the decisions they have not made
yet, propose the whole pipeline step by step with the source behind each step, and check their plan or schema
with review_plan. Reuse what an earlier topic already learned (notebook_read(query=…)).

Remember what was tried: when the user reports how something went (speed, quality, an error), log it with
notebook_write(section, experiment={change, settings, result, verdict}); before suggesting, read the topic's
experiments (notebook_read(section)) and never re-propose what already failed without saying what is different.

Answer like an expert: lead with the few strongest options, each with its measured gain (number, metric,
conditions, source), its cost, whether it needs training, code availability and how to plug it in. If nothing
measured beats the baseline, say so and give the experiment that would decide it.
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


def _unknown(source_id: str) -> dict[str, str]:
    return {"error": f"unknown source {source_id}: find it with the search tools first"}


# ---------------- finding ----------------

@mcp.tool()
def research_sweep(query: str, platforms: list[str] | None = None, per_platform: int = 6) -> dict:
    """Search every platform at once — start here when planning, choosing a model / library / approach, or entering
    an unfamiliar topic: papers, Hugging Face models, GitHub repos and issues, community (Hugging Face discussions of
    the matching models, Civitai checkpoints / LoRAs and ComfyUI nodes for generative topics, Hacker News), YouTube
    videos and the web (with a Tavily / Brave key; otherwise ready queries for your own web search), each graded. Then fetch the
    promising ones.

    Args:
        query: short English query, e.g. "talking head lip sync"
        platforms: subset of ["papers", "models", "repos", "issues", "community", "videos", "web"] (default: all), or community
            parts alone: "discussions", "civitai", "nodes", "hn"; ["web"] forces the built-in web search
        per_platform: results per platform (3-15)
    """
    from kb.research.sweep import research_sweep as _sweep

    return _sweep(query, platforms, per_platform)


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
    """Search Hugging Face models / datasets / spaces / papers: release date, downloads, likes, license, task,
    base model and relation (quantized / finetune / adapter). Matches repo names: use short name-like queries
    ("whisper russian"). For "what is newest for task X" use latest_models.

    Args:
        kind: models | datasets | spaces | papers;  sort: trending | downloads | likes | created | recent
    """
    from kb.research.platforms import hf_search as _hf

    kind = kind if kind in ("models", "datasets", "spaces", "papers") else "models"
    return _with_db(lambda db: _hf(db, query, kind, sort, min(limit, 30)))


@mcp.tool()
def find_papers(query: str | None = None, limit: int = 15, year_from: int | None = None,
                sources: list[str] | None = None, paper_id: str | None = None,
                direction: str = "references") -> dict:
    """Papers, live from OpenAlex, arXiv and Semantic Scholar plus the local cache; duplicates merged, ranked by
    agreement between sources and citations, each graded A-D. With paper_id instead of a query: walk its citation
    graph — direction "references" (what it builds on) or "citations" (follow-ups, reproductions, critiques).

    Args:
        query: focused English query (several narrow queries beat one broad one)
        sources: subset of ["openalex", "arxiv", "s2", "local"] (default: all; "s2" only with S2_API_KEY)
        paper_id: e.g. "arxiv:2211.17192" or "doi:10.1038/..."
    """
    if paper_id:
        from kb.research.graph import follow_citations

        return _with_db(lambda db: follow_citations(db, paper_id, direction, min(limit, 50)))
    if not query:
        return {"error": "pass query, or paper_id to follow citations"}
    from kb.research.find import SOURCES, default_sources
    from kb.research.find import find_papers as _find

    use = tuple(s for s in (sources or default_sources()) if s in SOURCES) or default_sources()
    return _with_db(lambda db: _find(db, query, min(limit, 40), year_from, use, on_paper=notebook.sections_with))


@mcp.tool()
def find_methods(goal: str, queries: list[str] | None = None, baselines: list[str] | None = None, years: int = 2,
                 limit: int = 12, section: str | None = None, save_top: int = 6) -> dict:
    """The strongest recent METHODS for a concrete goal ("a better sampler than X at the same cost"). Searches many
    phrasings plus the papers that cite the baseline to beat, ranks by meaning and evidence, and returns for each
    the claimed gain (numbers from the abstract), training-free or not, extra model calls, venue, code link.

    Args:
        goal: the goal with its constraints, one or two English sentences
        queries: 3-8 short English phrasings with different words for the same thing (method-family names, metric)
        baselines: the method(s) to beat, by name — newer work cites and compares to them
        section: a RESEARCH.md topic: the top `save_top` candidates are written there, marked as not yet verified
    """
    from kb.research.methods import find_methods as _find

    def run(db: Database) -> dict:
        out = _find(db, goal, queries, baselines, max(1, min(years, 6)), min(limit, 25))
        if section:
            tried = _safe(lambda: notebook.read_section(section)).get("experiments")
            if tried:
                out["already_tried"] = tried
                out["next"] = ("already_tried lists what the user ran in this topic: do not propose what failed or "
                               "did not help again unless the settings differ in a way that matters. ") + out["next"]
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
def study_topic(topic: str, task: str | None = None, terms: list[str] | None = None, kinds: list[str] | None = None,
                read: int = 10, section: str | None = None, depth: str = "normal") -> dict:
    """Learn a field like a specialist before advising on practical work in it (an annotation schema, a dataset,
    training, evaluation, a pipeline). Splits the topic into facets — foundations, reference datasets / systems,
    each concept in `terms`, the decisions every such project must make (from a checklist for the task kind),
    practices, pitfalls, tools — searches papers (surveys and canonical works welcome), web guides, forums and
    repos for each, reads the best `read` sources in full and returns the passages that answer each facet, plus
    the decisions to settle with the user. With `section`: whether its specialist knowledge is deep enough to
    advise (expert_ready) and what is missing.

    Args:
        topic: short English name of the field, e.g. "video action annotation"
        task: what the user is doing, in their words (picks the checklist)
        terms: 2-5 English key concepts to learn, e.g. ["atomic actions", "temporal action segmentation"]
        kinds: force checklists: annotation | training | evaluation | dataset | inference | generic
        read: sources to read in full now (0-15)
        section: the RESEARCH.md topic collecting this knowledge;  depth: quick | normal | deep
    """
    from kb.research.expertise import study_topic as _study

    depth = depth if depth in ("quick", "normal", "deep") else "normal"
    return _with_db(lambda db: _study(db, topic, task, terms, kinds, max(0, min(read, 15)), section, depth))


@mcp.tool()
def review_plan(plan: str, topic: str | None = None, section: str | None = None, kinds: list[str] | None = None,
                depth: str = "normal") -> dict:
    """Review the user's plan, annotation schema, label list or pipeline like a senior specialist: which decisions
    of the task-kind checklist it does not address, plus every pitfall, practice and decision learned in
    `section` (study_topic) to hold it against. Use it before building on a plan and whenever the user shares one.

    Args:
        plan: the plan / schema / pipeline as text (paste it whole)
        topic: what it is for;  section: the RESEARCH.md topic with the learned knowledge
    """
    from kb.research.expertise import review_plan as _review

    return _with_db(lambda db: _review(db, plan, topic, section, kinds, depth))


@mcp.tool()
def research_brief(topic: str, goal: str | None = None, environment: str | None = None, baseline: str | None = None,
                   success_metric: str | None = None, constraints: str | None = None, deliverable: str | None = None,
                   already_tried: str | None = None, user_declined: list[str] | None = None) -> dict:
    """Call FIRST for any research, planning or "find the best way to…" request. Fill every field you can from
    the conversation; it returns what is still missing as questions for the user. Ask them, call again with the
    answers, and search only when `ready` is true.

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
def source_info(source_id: str, critique: bool = False, code: bool = False, others_say: bool = False,
                stance: str | None = None) -> dict:
    """Everything about one source (paper, repo, issue, model, page): metadata, platform data, stored evidence
    grade, full-text status, notebook sections citing it. Options (each makes live requests):
      critique: re-grade A-D with reasons — papers: peer review, OpenReview status, venue, code, citations, vendor
        self-report; other sources: official org, adoption, maintenance, license, unsafe pickle weights, issues.
      code: for a paper — the official repo, project page and Hugging Face artifacts built on it.
      others_say: what later papers say about it (Semantic Scholar citation contexts), critical first;
        stance: critical | possible_critical | comparative | neutral.
    """
    def run(db: Database) -> dict:
        p = db.get_paper(source_id)
        if p is None and not code:
            return _unknown(source_id)
        from kb.research.find import brief

        out: dict = {}
        if p is not None:
            out = {**p, **brief(db, p["id"], abstract_chars=5000), "in_notebook": notebook.sections_with(p["id"])}
            if critique:
                out["critique"] = _critique(db, p)
            if others_say:
                from kb.critic.citation_contexts import what_others_say

                out["others_say"] = what_others_say(db, p["id"], stance, 20)
        if code:
            from kb.research.ml import paper_code

            out["code"] = paper_code((p or {}).get("arxiv_id") or source_id)
        return out

    return _with_db(run)


def _critique(db: Database, p: dict) -> dict:
    if p.get("kind", "paper") != "paper":
        import json

        from kb.critic.sources import grade_doc

        g = grade_doc(p["kind"], p.get("extra") or {}, p["id"])
        db.conn.execute("INSERT OR REPLACE INTO critique (source_id, kind, grade, reasons, flags) VALUES (?, ?, ?, ?, ?)",
                        (p["id"], p["kind"], g["grade"], json.dumps(g["reasons"]), json.dumps(g["flags"])))
        db.commit()
        return g
    from kb.critic.openreview import check_openreview
    from kb.critic.scoring import grade_and_save

    try:
        check_openreview(db, paper_ids=[p["id"]])
    except Exception:
        pass
    r = grade_and_save(db, p["id"])
    db.commit()
    return r or {}


# ---------------- reading ----------------

@mcp.tool()
def fetch(url: str | None = None, urls: list[str] | None = None, paper_ids: list[str] | None = None,
          refresh: bool = False) -> dict:
    """Read sources and cache them for search_fulltext. url / urls (up to 12, read in parallel): any page — docs,
    blogs, PDFs, GitHub repos (README + release notes), issue / PR threads (maintainers marked), Hugging Face
    model cards and discussions, Civitai models, YouTube videos (transcript with timestamps), arXiv papers, Hacker News threads; returns the text (12k chars, 5k each for several), an outline
    and the source id for notebook_add_source. paper_ids (up to 10, from find_papers): full texts from arXiv or an
    open-access PDF, split into passages with their numbers extracted."""
    out: dict = {}
    if paper_ids:
        from kb.process.fulltext import fetch_papers

        out["papers"] = _with_db(lambda db: fetch_papers(db, paper_ids[:10], refresh=refresh))
    pages = (urls or []) + ([url] if url else [])
    if len(pages) > 1:
        from kb.research.web import fetch_many

        out["results"] = fetch_many(pages)
    elif pages:
        from kb.research.web import fetch_url

        def run(db: Database) -> dict:
            try:
                return fetch_url(db, pages[0])
            except Exception as e:
                return {"url": pages[0], "error": f"{type(e).__name__}: {str(e)[:200]}",
                        "next": "try your own fetch tool, or another page with the same information"}

        page = _with_db(run)
        if not paper_ids:
            return page
        out["page"] = page
    return out or {"error": "pass url, urls or paper_ids"}


@mcp.tool()
def search_fulltext(
    query: str | None = None,
    k: int = 8,
    section: str | None = None,
    paper_ids: list[str] | None = None,
    year_from: int | None = None,
    grades: str | None = None,
    kinds: list[str] | None = None,
    chunk_id: int | None = None,
    window: int = 1,
) -> list[dict] | dict:
    """Keyword + semantic search INSIDE everything fetched so far (methods, tables, ablations, limitations). Write
    the query in English. Every hit has a citation (paper, section), the paper's grade and a chunk_id; pass
    chunk_id (instead of a query) to read that passage with `window` neighbours on each side before citing it.

    Args:
        query: a natural-language English question, e.g. "how does acceptance rate change with draft model size"
        section: limit to papers of this notebook section;  paper_ids: limit to these sources
        grades: e.g. "AB" to keep only strong evidence
        kinds: subset of ["text", "table", "figure"]; tables carry the actual numbers
    """
    from kb.process.search import context, search

    if chunk_id is not None:
        return _with_db(lambda db: context(db, chunk_id, max(0, min(window, 4))))
    if not query:
        return {"error": "pass query, or chunk_id to read a passage"}

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
def evidence(
    claim: str | None = None,
    section: str | None = None,
    kind: str | None = None,
    hardware: str | None = None,
    model: str | None = None,
    text: str | None = None,
    paper_id: str | None = None,
    limit: int = 30,
) -> dict | list[dict]:
    """Check numbers before trusting them. With a claim ("4-bit quantization loses <1% on MMLU"): related papers
    with grades, full-text passages and extracted numbers, and a verdict on the strength of the evidence. Without
    one: the numbers extracted from fetched full texts, verbatim with their conditions — speedups ("2.3x on
    H100"), memory, step counts, latency, benchmark scores — to compare results across papers.

    Args:
        section: look only at papers of this notebook section (with a claim)
        kind: speedup | memory | steps | latency | quality_metric
        hardware: e.g. "H100";  model: e.g. "Llama";  text: words that must appear;  paper_id: one paper
    """
    if claim:
        from kb.critic.claims import verify_claim

        def verify(db: Database) -> dict:
            try:
                return verify_claim(db, claim, _section_papers(section))
            except notebook.NotebookError as e:
                return {"error": str(e)}

        return _with_db(verify)
    from kb.process.claims import find

    def numbers(db: Database) -> list[dict]:
        pid = (db.get_paper(paper_id) or {"id": paper_id})["id"] if paper_id else None
        return find(db, kind, hardware, model, text, pid, min(limit, 100))

    return _with_db(numbers)


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
    """Are these current? PyPI packages ("vllm==0.8.1", "torch"), Hugging Face models (newer models of the same
    family and newer alternatives) and GitHub repos ("gh:vllm-project/vllm@v0.9.0"). For outdated pinned versions
    it digests the release notes in between: breaking changes, and lines about `focus` (e.g. "Qwen3-Omni audio")."""
    from kb.research.versions import check_versions as _check

    return _check(items, focus)


# ---------------- choosing models ----------------

@mcp.tool()
def latest_models(task: str, months: int = 9, limit: int = 10) -> dict:
    """What is new for a task: the newest strong base models (labs' own releases first), the hottest derivatives
    (fine-tunes, quantizations, LoRAs) and older popular ones for comparison, with release dates, plus
    leaderboards. Call before recommending any model.

    Args:
        task: free text or a Hugging Face task, e.g. "text-to-video", "speech recognition", "LLM", "TTS"
        months: how recent counts as new
    """
    from kb.research.models import latest_models as _latest

    return _latest(task, max(1, min(months, 36)), min(limit, 25))


@mcp.tool()
def model_variants(model_id: str, vram_gb: float | None = None, gpu: str | None = None,
                   compare_with: list[str] | None = None) -> dict:
    """Which weights to download: every precision in the model's repo (bf16 / fp8 / int8 / GGUF …) and the
    community's quantized versions, with sizes, whether each fits the GPU (detected, or pass vram_gb / gpu) and
    the quality cost; recommends the best quality that fits and says whether the GPU computes FP8 / FP4 natively.
    compare_with: up to 5 other models for a side-by-side table (release date, parameters, architecture, context,
    license, downloads, benchmark tables from the cards — self-reported, compare matching settings only).

    Args:
        model_id: Hugging Face repo, e.g. "Qwen/Qwen3-8B"
    """
    from kb.research.models import model_variants as _variants

    try:
        out = _variants(model_id, vram_gb, gpu)
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    if compare_with:
        from kb.research.ml import compare_models

        out["comparison"] = compare_models([model_id, *compare_with[:5]])
    return out


@mcp.tool()
def estimate_resources(model_id: str, mode: str = "inference", precision: str | None = None, seq_len: int = 4096,
                       batch: int = 1, lora_rank: int = 16, gradient_checkpointing: bool = True,
                       train_tokens: float | None = None, gpu: str | None = None, vram_gb: float | None = None,
                       num_gpus: int = 1, kv_cache: str = "bf16") -> dict:
    """Memory and time for running or training a Hugging Face model, from its real config: VRAM breakdown
    (weights, KV cache, activations, gradients, optimizer states), whether it fits the GPU (detected, or pass
    gpu / vram_gb), decode tokens/s, training hours for train_tokens, and what to change if it does not fit.

    Args:
        mode: inference | lora | qlora | full
        precision: bf16 | fp16 | fp8 | int8 | q8 | q6 | q5 | q4 | nf4 | awq | gptq | fp4 (default bf16; nf4 for qlora)
        train_tokens: total training tokens (dataset tokens x epochs) for a time estimate
        num_gpus: tensor parallel (inference) or FSDP / ZeRO-3 (full);  kv_cache: bf16 | fp8
    """
    from kb.research.ml import estimate_resources as _est

    return _est(model_id, mode, precision, max(1, seq_len), max(1, batch), max(1, lora_rank), gradient_checkpointing,
                train_tokens, gpu, vram_gb, max(1, num_gpus), kv_cache)


@mcp.tool()
def dataset_info(dataset_id: str, sample_rows: int = 3) -> dict:
    """A Hugging Face dataset before using it: configs and splits with row counts, size, columns and types, sample
    rows, license, gating, and the most used models trained on it."""
    from kb.research.ml import dataset_info as _ds

    return _ds(dataset_id, max(0, min(sample_rows, 10)))


# ---------------- RESEARCH.md ----------------

@mcp.tool()
def notebook_read(section: str | None = None, query: str | None = None, refresh: bool = False,
                  since: str | None = None) -> dict | list[dict]:
    """Read RESEARCH.md. No arguments: topics (question, source count, last update) and the local cache size.
    section: one topic in full. query: where earlier research mentions these words. section + refresh=true: what
    appeared since the topic was last updated (or since the ISO date `since`) — its saved queries re-run for
    papers, models, repos and issues, and the models / repos it cites checked for newer versions."""
    if query:
        return notebook.search(query)
    if section and refresh:
        from kb.research.refresh import refresh as _refresh

        return _safe(lambda: _refresh(section, since))
    if section:
        return _safe(lambda: notebook.read_section(section))
    from kb.research.refresh import stale_sections

    out = {**notebook.overview(), "local_base": _with_db(lambda db: db.stats())}
    stale = stale_sections()
    if stale:
        out["stale"] = stale
        out["hint"] = "Sections older than 30 days: notebook_read(section, refresh=true) before relying on them."
    return out


@mcp.tool()
def notebook_write(
    section: str | None = None,
    title: str | None = None,
    question: str | None = None,
    summary: str | None = None,
    append: bool = False,
    add_queries: list[str] | None = None,
    remove_papers: list[str] | None = None,
    knowledge: list[dict] | None = None,
    experiment: dict | None = None,
) -> dict:
    """Open or update a topic in RESEARCH.md. title + question: open a topic (title short, in the user's language;
    an existing topic with the same title is returned to build on). section: update its conclusions (summary,
    Markdown; append=true adds to them), question, search queries, or remove sources that turned out irrelevant.
    knowledge: what a specialist must know, learned from sources — [{"kind": concept | decision | practice |
    pitfall | metric | tool, "text": "…", "sources": ["arxiv:…", "web:…"]}]; one fact per item, in the user's
    language, with what it means for the work (a decision lists its options and when to pick each).
    experiment: log one try the user (or you) ran — {"change": "what was tried", "settings": "…", "result": "what
    came out, with numbers", "verdict": better | worse | same | failed | unclear}; read them back before suggesting
    anything so nothing that already failed is proposed again."""
    if title:
        opened = _safe(lambda: notebook.start(title, question or "", add_queries))
        if "error" in opened or not (summary or remove_papers or knowledge or experiment):
            return opened
        section, question, add_queries = opened["section"], None, None
    if not section:
        return {"error": "pass title (+ question) to open a topic, or section to update one"}
    out: dict = {}
    if experiment:
        e = experiment
        out["experiment"] = _safe(lambda: notebook.add_experiment(
            section, str(e.get("change", "")), str(e.get("result", "")), str(e.get("verdict", "")),
            str(e.get("settings", "")), e.get("date")))
        if "error" in out["experiment"] or not (summary is not None or question or add_queries or remove_papers
                                                 or knowledge):
            return out["experiment"]
    if summary is not None or question or add_queries or remove_papers:
        out = {**out, **_safe(lambda: notebook.write(section, summary, append, question, add_queries, remove_papers))}
        if "error" in out:
            return out
    if knowledge:
        def check(db: Database) -> tuple[list[dict], list[str]]:
            good, unknown = [], []
            for it in knowledge:
                if not isinstance(it, dict):
                    continue
                ids = []
                for i in it.get("sources") or []:
                    p = db.get_paper(str(i))
                    (ids.append(p["id"]) if p else unknown.append(str(i)))
                good.append({**it, "sources": ids})
            db.touch({i for it in good for i in it["sources"]})
            db.commit()
            return good, unknown

        items, unknown = _with_db(check)
        res = _safe(lambda: notebook.add_knowledge(section, items))
        if unknown:
            res["unknown_sources"] = sorted(set(unknown))
            res["note"] = "Only sources found through these tools can be cited; unknown ids were dropped."
        return {**out, "knowledge": res} if out else res
    return out or {"error": "nothing to change: pass summary, question, add_queries, remove_papers, knowledge or "
                            "experiment"}


@mcp.tool()
def notebook_add_source(section: str, source_id: str, notes: str, replace: bool = False) -> dict:
    """Add a source (paper, repo, issue, model, page, thread) to a topic. Its title line (link, type, year, grade,
    stars / downloads / citations, license, state) is generated; `notes` is your Markdown analysis:
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
        db.touch([p["id"]])
        db.commit()
        return _safe(lambda: notebook.add_paper(section, {**p, "grade": grade}, notes, replace))

    return _with_db(run)


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
        "1. notebook_read(query=…) to reuse earlier research; notebook_write(title=…, question=…) to open the topic "
        "(include the brief in the question).\n"
        "2. If this is practical work in a field (annotation, dataset, training, evaluation, a pipeline), study "
        "the field first: study_topic(topic, terms=[key concepts], section=…), read, write "
        "notebook_write(section, knowledge=[…]) and repeat until expert_ready; check the user's plan or schema "
        "with review_plan. research_sweep for the landscape. If the user wants a solution or an improvement over something, "
        "find_methods(goal, queries=[3-8 phrasings with different words], baselines=[what to beat]). Then focused "
        "find_papers / github_search (repos and issues) / hf_search / research_sweep(platforms=[\"web\"]). Use "
        "your own web search too and fetch the useful pages."
        + (" find_papers(paper_id=…) on the central papers in both directions." if depth == "deep" else "") + "\n"
        "3. Read before concluding: fetch(url=…) for repos, model cards, docs, issue threads; fetch(paper_ids=…) for "
        "papers; then search_fulltext(English question, section=…) and evidence(kind=…) for exact numbers and "
        "conditions.\n"
        "4. For model choices: latest_models for the task (state release dates) and model_variants for the user's "
        "GPU (FP8 / INT8 before 4-bit when bf16 does not fit); check_versions on every library you will recommend; "
        + ("source_info(critique=true, others_say=true) for every key source and evidence(claim=…) for every key "
           "claim; " if depth == "deep" else "source_info(critique=true) on the sources your conclusions rely on; ")
        + "say when evidence is only C/D.\n"
        "5. notebook_add_source per source (what it says, numbers with conditions and where they come from, "
        "limitations, applicability), then notebook_write(section, summary=…) with conclusions and a concrete "
        "recommendation, citing source ids.\n"
        "Advise as a specialist: explain the concepts the user needs, raise the decisions they have not made, and "
        "propose the full pipeline step by step with the source behind each step.\n"
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
        "2. Read the best matching issue / PR in full (fetch(url=…)), including maintainers' comments; note the "
        "version that contains the fix and any documented workaround.\n"
        "3. check_versions for the libraries and models involved; if outdated, read the release notes first.\n"
        "4. If nothing matches, use your own web search with the exact message, then fetch the results.\n"
        "5. Apply the documented fix, verify it, and if it took effort to find, record it in RESEARCH.md "
        "(notebook_write(title=…) / notebook_add_source / notebook_write(section, summary=…)).\n\n"
        f"Error:\n{error}"
    )


def main() -> None:
    import logging

    from kb.update import update_and_restart

    update_and_restart()   # new version published? fast-forward and re-exec before anything starts

    # Agents show the server's stderr: keep it to warnings, not one line per HTTP request.
    logging.basicConfig(level=logging.WARNING)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:   # background helper: the user's own programs get the CPU first
        os.nice(int(os.environ.get("KB_NICE", "10")))
    except (AttributeError, OSError, ValueError):
        pass
    import threading

    from kb.store.prune import prune_daily

    # Nothing is loaded at start: the embedding model loads on the first search and unloads when idle.
    threading.Thread(target=prune_daily, name="prune", daemon=True).start()
    mcp.run()


if __name__ == "__main__":
    main()
