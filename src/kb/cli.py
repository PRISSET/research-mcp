"""Command line interface: `uv run kb --help`."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from kb.logging_setup import setup_logging
from kb.store import Database

app = typer.Typer(help="Research MCP: papers, full texts, evidence grades, RESEARCH.md", no_args_is_help=True)
base_app = typer.Typer(help="Maintenance of the local paper base (batch jobs)", no_args_is_help=True)
app.add_typer(base_app, name="base")
console = Console()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    setup_logging(verbose)


@app.command()
def find(query: str, limit: int = 15, year_from: int = typer.Option(None)) -> None:
    """Search papers live (OpenAlex, arXiv, Semantic Scholar + local cache)."""
    from kb.research.find import find_papers

    with Database() as db:
        res = find_papers(db, query, limit, year_from)
    console.print(res["sources"])
    table = Table("id", "year", "grade", "cit.", "title")
    for r in res["results"]:
        table.add_row(r["id"][:40], str(r["year"] or ""), r["grade"] or "", str(r["citations"] or 0), (r["title"] or "")[:80])
    console.print(table)


@app.command()
def update(check_only: bool = typer.Option(False, "--check", help="only report whether an update exists")) -> None:
    """Update research-mcp to the published version (the server also does this on every start)."""
    from kb.update import check

    console.print(check(apply=not check_only))


@app.command()
def fetch(paper_ids: list[str]) -> None:
    """Download and index full texts of these papers."""
    from kb.process.fulltext import fetch_papers

    with Database() as db:
        console.print(fetch_papers(db, paper_ids))


@app.command()
def search(query: str, k: int = 8) -> None:
    """Hybrid search inside fetched full texts (BM25 + vectors)."""
    from kb.process.search import search as fulltext_search

    with Database() as db:
        for r in fulltext_search(db, query, k=k):
            console.print(f"[bold]{r['score']:.3f}[/] {r['title'][:70]} [{r['grade']}] — {r['section'][:60]}")
            console.print("   " + r["text"][:300].replace("\n", " "))


keys_app = typer.Typer(help="Optional API keys (stored in ~/.config/research-mcp/keys.env)", invoke_without_command=True)
app.add_typer(keys_app, name="keys")


@keys_app.callback()
def keys_list(ctx: typer.Context) -> None:
    """Which keys are set, what each improves and where to get it."""
    if ctx.invoked_subcommand:
        return
    from kb import keys

    t = Table("key", "set", "improves", "get it at")
    for k in keys.status():
        t.add_row(k["name"], "[green]yes[/]" if k["set"] else "-", k["improves"], k["get"])
    console.print(t)
    console.print(f"File: {keys.path()}  ·  add: uv run kb keys set NAME VALUE  ·  then restart the agent session")


@keys_app.command("set")
def keys_set(name: str, value: str = typer.Argument("", help="empty removes the key")) -> None:
    from kb import keys

    if name not in keys.NAMES:
        console.print(f"[yellow]{name} is not a known key; stored anyway[/]")
    console.print(f"{'saved' if value else 'removed'}: {name} in {keys.set_key(name, value)} — restart the agent session")


@app.command()
def stats() -> None:
    """Size of the local base and the notebook."""
    from kb import notebook

    with Database() as db:
        console.print(db.stats())
    console.print(notebook.overview())


@base_app.command("citations")
def base_citations(limit: int = typer.Option(None, help="Max papers to enrich")) -> None:
    """Fetch citation counts / TL;DRs from Semantic Scholar for papers not checked yet."""
    from kb.collect.citations import enrich_citations

    with Database() as db:
        checked, found = enrich_citations(db, limit)
    console.print(f"checked {checked}, found in S2 {found}")


@base_app.command("openreview")
def base_openreview(limit: int = 300) -> None:
    """Check OpenReview acceptance status for papers not checked yet."""
    from kb.critic.openreview import check_openreview

    with Database() as db:
        console.print(check_openreview(db, limit))


@base_app.command("grade")
def base_grade() -> None:
    """Recompute evidence grades A-D for all papers."""
    from kb.critic.scoring import grade_all

    with Database() as db:
        console.print(grade_all(db))


@base_app.command("fulltext")
def base_fulltext(
    limit: int = typer.Option(None, help="Max papers this run"),
    grades: str = typer.Option("ABC", help="Critique grades to include"),
    interval: float = typer.Option(1.0, help="Seconds between arXiv requests"),
    retry_failed: bool = typer.Option(False),
) -> None:
    """Fetch full texts for all arXiv papers that have none yet (resumable)."""
    from kb.process import fulltext

    with Database() as db:
        console.print(fulltext.collect(db, limit, grades, interval, retry_failed))
        console.print(fulltext.stats(db))


@base_app.command("embed")
def base_embed(limit: int = typer.Option(None)) -> None:
    """Embed passages that have no vector yet."""
    from kb.process import embed

    with Database() as db:
        console.print(embed.embed_pending(db, limit=limit))


@base_app.command("prune")
def base_prune(days: int = typer.Option(60, help="Drop full texts not cited and not used for this many days")) -> None:
    """Drop what the agent read but never used again (the notebook's sources are always kept), then VACUUM."""
    from kb.store.prune import prune

    with Database() as db:
        console.print(prune(db, days))


@base_app.command("claims")
def base_claims(rebuild: bool = typer.Option(False)) -> None:
    """Extract numbers (speedups, memory, steps, latency, benchmark scores) from passages."""
    from kb.process import claims

    with Database() as db:
        console.print(claims.extract_all(db, rebuild=rebuild))


@base_app.command("contexts")
def base_contexts(limit: int = 200) -> None:
    """Fetch citation contexts (what other papers say) for A/B papers."""
    from kb.critic.citation_contexts import collect

    with Database() as db:
        console.print(collect(db, limit))


if __name__ == "__main__":
    app()
