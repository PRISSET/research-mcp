"""Command line interface: `uv run kb --help`."""

from __future__ import annotations

from pathlib import Path

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


searxng_app = typer.Typer(help="Local SearXNG for free, keyless web search (no Docker)", no_args_is_help=True)
app.add_typer(searxng_app, name="searxng")


@searxng_app.command("install")
def searxng_install() -> None:
    """Install (or update) SearXNG in ~/.local/share/research-mcp/searxng and point research-mcp at it."""
    from kb import searxng

    how = searxng.install()
    console.print(f"[green]SearXNG ready[/] at {searxng.url()} — {how}. SEARXNG_URL saved; restart the agent session.")


@searxng_app.command("uninstall")
def searxng_uninstall() -> None:
    from kb import searxng

    searxng.uninstall()
    console.print("SearXNG removed, SEARXNG_URL cleared")


opencode_app = typer.Typer(help="Register research-mcp in OpenCode", no_args_is_help=True)
app.add_typer(opencode_app, name="opencode")


@opencode_app.command("register")
def opencode_register(name: str = "research", notebook: str = "~/research/RESEARCH.md", lang: str = "ru") -> None:
    """Add the server to OpenCode's global config (~/.config/opencode/opencode.json), keeping everything else."""
    from kb import opencode
    from kb.config import ROOT

    env = {"RESEARCH_NOTEBOOK": str(Path(notebook).expanduser()), "RESEARCH_LANG": lang, "CUDA_VISIBLE_DEVICES": ""}
    where = opencode.register(name, str(ROOT), env)
    console.print(f"[green]Registered '{name}'[/] in {where}. Restart OpenCode; `opencode mcp list` shows it.")


@opencode_app.command("unregister")
def opencode_unregister(name: str = "research") -> None:
    from kb import opencode

    console.print("removed" if opencode.unregister(name) else f"'{name}' is not in {opencode.config_path()}")


redlib_app = typer.Typer(help="Local Redlib for reading Reddit threads and search (no Docker)", no_args_is_help=True)
app.add_typer(redlib_app, name="redlib")


@redlib_app.command("install")
def redlib_install() -> None:
    """Build (or update) Redlib with cargo in ~/.local/share/research-mcp/redlib and point research-mcp at it."""
    from kb import redlib

    how = redlib.install()
    console.print(f"[green]Redlib ready[/] at {redlib.url()} — {how}. REDLIB_URL saved; restart the agent session.")


@redlib_app.command("uninstall")
def redlib_uninstall() -> None:
    from kb import redlib

    redlib.uninstall()
    console.print("Redlib removed, REDLIB_URL cleared")


guard_app = typer.Typer(help="Code guard: task ledger + Claude Code hooks (open items on every prompt, no "
                             "finishing while items are open or code is unreviewed)", no_args_is_help=True)
app.add_typer(guard_app, name="guard")


@guard_app.command("install")
def guard_install() -> None:
    """Add the hooks to ~/.claude/settings.json (other hooks and settings stay; a backup is written)."""
    from kb import guard

    console.print(f"[green]Code guard hooks added[/] to {guard.install()}. They apply to new Claude Code sessions.")


@guard_app.command("uninstall")
def guard_uninstall() -> None:
    from kb import guard

    console.print("hooks removed" if guard.uninstall() else f"no code-guard hooks in {guard.settings_path()}")


@guard_app.command("show")
def guard_show(project: str = typer.Argument(".", help="repository path")) -> None:
    """The task ledger of a project."""
    from kb import guard

    led = guard.update(str(Path(project).resolve()), show_all=True)
    for i in led["items"]:
        console.print(guard._line(i) + f"  [{i['status']}]")
    if not led["items"]:
        console.print(f"empty ledger for {led['project']}")


pack_app = typer.Typer(help="Knowledge packs: trainer docs, issues and training recipes of one field, "
                            "updated daily in parts", no_args_is_help=True)
app.add_typer(pack_app, name="pack")


def _print_report(report: dict) -> None:
    for name, r in report.items():
        if r.get("error"):
            console.print(f"[yellow]{name}[/]: {r['error']}")
        elif r.get("status"):
            console.print(f"{name}: v{r['version']} {r['status']}")
        else:
            console.print(f"[green]{name}[/]: v{r['from']} -> v{r['to']} ({r['downloaded']}, "
                          f"{r['bytes'] / 1e6:.1f} MB): {r['written']} written, {r['deleted']} deleted, "
                          f"{r['unchanged']} unchanged; vectors: {r['vectors']}")


@pack_app.command("list")
def pack_list() -> None:
    """Installed and published packs."""
    from kb.packs import client

    s = client.status()
    t = Table("pack", "installed", "published", "documents", "recipes", "download", "about")
    avail = s.get("available") if isinstance(s.get("available"), dict) and "error" not in s["available"] else {}
    for name in sorted(set(avail) | set(s["installed"])):
        a, i = avail.get(name) or {}, s["installed"].get(name)
        t.add_row(name, f"v{i['version']}" if i else "-", f"v{a['version']}" if a else "-", str(a.get("docs", "")),
                  str(a.get("recipes", "")), f"{a['bytes'] / 1e6:.0f} MB" if a else "", (a.get("title") or "")[:60])
    console.print(t)
    if isinstance(s.get("available"), dict) and s["available"].get("error"):
        console.print(f"[yellow]{s['available']['error']}[/]")


@pack_app.command("install")
def pack_install(names: list[str]) -> None:
    """Download packs (the snapshot once; later updates download only the changes)."""
    from kb.packs import client

    _print_report(client.update(names, install=True))


@pack_app.command("update")
def pack_update(names: list[str] = typer.Argument(None)) -> None:
    """Bring installed packs up to date (the server also does this daily)."""
    from kb.packs import client

    _print_report(client.update(names or None))


@pack_app.command("remove")
def pack_remove(name: str) -> None:
    from kb.packs import client

    console.print(client.remove(name))


@pack_app.command("build")
def pack_build(names: list[str] = typer.Argument(None, help="packs to build (default: all defined)"),
               out: Path = typer.Option(Path("dist/packs"), help="where the new files and manifest.json go"),
               base: str = typer.Option(None, help="published state to diff against: release URL or folder "
                                                   "(default: KB_PACKS_URL or the GitHub release)")) -> None:
    """Build new pack versions (GitHub Actions runs this daily). Publish: upload `out` to the release."""
    import json as _json

    from kb.packs import build

    report = build.build(names or build.available(), out, base)
    console.print_json(_json.dumps(report, default=str))
    if any(r.get("error") for r in report.values()):
        raise typer.Exit(1)   # red in GitHub Actions instead of a "successful" empty pack


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
