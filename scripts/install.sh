#!/usr/bin/env bash
# One-command install of the research MCP server and its registration in Claude Code, Codex CLI or OpenCode.
#
#   curl -fsSL https://raw.githubusercontent.com/PRISSET/research-mcp/main/scripts/install.sh | bash -s -- --agent claude
#   scripts/install.sh [--agent claude|codex|opencode|all] [--dir PATH] [--notebook FILE] [--lang ru|en] [--embed|--no-embed]
#                      [--agent-rules] [--no-register] [--searxng] [--redlib]
#
#   --agent NAME     which agent to register in (and write rules for): claude, codex, opencode or all. Default: the agent
#                    running this script (Claude Code sets CLAUDECODE, Codex CODEX_THREAD_ID / CODEX_SANDBOX*);
#                    run by hand in a terminal, every installed agent
#   --dir PATH       where to clone (default: this checkout, or ~/research-mcp)
#   --notebook FILE  the shared research notebook (default: ~/research/RESEARCH.md)
#   --lang ru|en     language of the notebook headings (default: ru)
#   --embed          semantic search: a static embedding model (model2vec, no torch, ~130 MB RAM) (default)
#   --no-embed       skip it: keyword search only
# The server runs on the CPU (registered with CUDA_VISIBLE_DEVICES=""): the GPU stays free for your own work.
#   --searxng        install a local SearXNG (no Docker) for free, keyless web search, as a systemd user service
#   --redlib         build a local Redlib (needs Rust's cargo) so Reddit threads and search can be read
#   --agent-rules    add a short rules block to the agent's global instructions (~/.claude/CLAUDE.md for Claude Code,
#                    ~/.codex/AGENTS.md for Codex, ~/.config/opencode/AGENTS.md for OpenCode) so it looks models / versions / errors up with this MCP in every
#                    project (marked block; remove: --remove-agent-rules)
#   --no-register    do not touch Claude Code / Codex CLI / OpenCode configs
# Safe to re-run: pulls updates, re-syncs, re-registers. No API keys needed.
set -euo pipefail

REPO_URL="https://github.com/PRISSET/research-mcp.git"
NAME="research"
SEARXNG=0 REDLIB=0 DIR="" NOTEBOOK="$HOME/research/RESEARCH.md" LANG_NB="ru" EMBED=1 REGISTER=1 RULES=keep AGENT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --agent) AGENT="$2"; shift ;;
    --dir) DIR="$2"; shift ;;
    --notebook) NOTEBOOK="$2"; shift ;;
    --lang) LANG_NB="$2"; shift ;;
    --embed) EMBED=1 ;;
    --no-embed) EMBED=0 ;;
    --no-register) REGISTER=0 ;;
    --agent-rules) RULES=install ;;
    --searxng) SEARXNG=1 ;;
    --redlib) REDLIB=1 ;;
    --remove-agent-rules) RULES=remove ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*" >&2; }

# --- which agent: the one running this script, unless --agent says otherwise
if [ -z "$AGENT" ]; then
  if [ -n "${CLAUDECODE:-}" ]; then AGENT=claude
  elif [ -n "${CODEX_THREAD_ID:-}${CODEX_SANDBOX:-}${CODEX_SANDBOX_NETWORK_DISABLED:-}" ]; then AGENT=codex
  elif [ -n "${OPENCODE:-}${OPENCODE_SESSION_ID:-}" ]; then AGENT=opencode
  else AGENT=all
  fi
fi
case "$AGENT" in
  claude|codex|opencode|all) ;;
  *) echo "--agent must be claude, codex, opencode or all (got: $AGENT)" >&2; exit 2 ;;
esac
want() { [ "$AGENT" = all ] || [ "$AGENT" = "$1" ]; }

# --- where the checkout lives
if [ -z "$DIR" ]; then
  here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd || true)"
  if [ -n "$here" ] && grep -q 'research-mcp' "$here/pyproject.toml" 2>/dev/null; then DIR="$here"; else DIR="$HOME/research-mcp"; fi
fi

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
if [ -d "$DIR/.git" ]; then
  say "Updating $DIR"
  if ! git -C "$DIR" pull --ff-only; then
    # The published history may have been rewritten: move a clean checkout over, keeping the old state in a branch.
    if git -C "$DIR" fetch origin main && [ -z "$(git -C "$DIR" status --porcelain --untracked-files=no)" ] \
       && ! git -C "$DIR" merge-base HEAD origin/main >/dev/null 2>&1; then
      git -C "$DIR" branch -f "before-update-$(date +%Y%m%d)" HEAD
      git -C "$DIR" reset --hard origin/main && say "Moved to the new published history (old state kept in a branch)"
    else
      warn "git pull failed, using the current checkout"
    fi
  fi
else
  say "Cloning into $DIR"
  git clone "$REPO_URL" "$DIR"
fi
cd "$DIR"

# --- uv (Python 3.12 comes with it)
if ! command -v uv >/dev/null; then
  say "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

extras=(--extra fulltext)
[ "$EMBED" = 1 ] && extras+=(--extra embed)
say "Installing dependencies (${extras[*]})"
uv sync "${extras[@]}"

# --- smoke test: the server starts and lists its tools
say "Checking the MCP server"
limit=""; command -v timeout >/dev/null && limit="timeout 60"
tools=$( (printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"install","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'; sleep 10) \
  | $limit uv run research-mcp 2>/dev/null \
  | uv run python -c 'import sys,json
for l in sys.stdin:
    m = json.loads(l)
    if m.get("id") == 2: print(len(m["result"]["tools"]))' || true)
if [ -n "$tools" ]; then echo "OK: $tools tools"; else warn "the server did not answer tools/list — run: uv run research-mcp"; fi

# --- register in the agents that are installed
registered=()
envs=(RESEARCH_NOTEBOOK="$NOTEBOOK" RESEARCH_LANG="$LANG_NB" CUDA_VISIBLE_DEVICES=)
if [ "$REGISTER" = 1 ]; then
  if want claude && command -v claude >/dev/null; then
    say "Registering in Claude Code (user scope)"
    for old in "$NAME" h3-kb; do claude mcp remove "$old" -s user >/dev/null 2>&1 || true; done
    claude mcp add "$NAME" -s user -e "${envs[0]}" -e "${envs[1]}" -e "${envs[2]}" -- uv --directory "$DIR" run research-mcp \
      && registered+=("Claude Code")
  fi
  if want codex && command -v codex >/dev/null; then
    say "Registering in Codex CLI"
    codex mcp remove h3-kb >/dev/null 2>&1 || true
    codex mcp add "$NAME" --env "${envs[0]}" --env "${envs[1]}" --env "${envs[2]}" -- uv --directory "$DIR" run research-mcp \
      && registered+=("Codex CLI")
    cfg="${CODEX_HOME:-$HOME/.codex}/config.toml"
    # `codex mcp add` rewrites the table, so the timeouts go back in every time
    if [ -f "$cfg" ]; then
      awk -v t="[mcp_servers.$NAME]" '{print} $0==t {print "startup_timeout_sec = 120"; print "tool_timeout_sec = 300"}' \
        "$cfg" > "$cfg.tmp" && mv "$cfg.tmp" "$cfg"
    fi
  fi
  if want opencode && command -v opencode >/dev/null; then
    say "Registering in OpenCode (global config)"
    (cd "$DIR" && uv run kb opencode register --name "$NAME" --notebook "$NOTEBOOK" --lang "$LANG_NB") \
      && registered+=("OpenCode")
  fi
fi

# --- optional: global agent rules (Claude Code reads ~/.claude/CLAUDE.md, Codex ~/.codex/AGENTS.md, OpenCode
#     ~/.config/opencode/AGENTS.md or, without it, ~/.claude/CLAUDE.md)
if [ "$RULES" != keep ]; then
  say "Agent rules: $RULES"
  targets=()
  if want claude && command -v claude >/dev/null; then targets+=("$HOME/.claude/CLAUDE.md"); fi
  if want codex && command -v codex >/dev/null; then targets+=("${CODEX_HOME:-$HOME/.codex}/AGENTS.md"); fi
  if want opencode && command -v opencode >/dev/null; then
    # OpenCode reads ~/.claude/CLAUDE.md only while its own AGENTS.md is missing: same file, no second copy
    oc=$(cd "$DIR" && uv run python -c 'from kb.opencode import rules_path; print(rules_path())')
    case " ${targets[*]} " in *" $oc "*) ;; *) targets+=("$oc") ;; esac
  fi
  if [ ${#targets[@]} -gt 0 ]; then
    uv run python -m kb.rules "$RULES" "${targets[@]}"
  else
    warn "no agent found for --agent $AGENT: no rules written"
  fi
fi

if [ "$SEARXNG" = 1 ]; then
  say "Installing SearXNG (local web search)"
  (cd "$DIR" && uv run kb searxng install) || warn "SearXNG install failed: web search falls back to the agent's own"
fi

if [ "$REDLIB" = 1 ]; then
  say "Building Redlib (Reddit reader, a few minutes)"
  (cd "$DIR" && uv run kb redlib install) || warn "Redlib install failed: Reddit is searched with the agent's own web search"
fi

say "Done"
echo "Installed in: $DIR"
echo "Notebook:     $NOTEBOOK (created on the first research)"
if [ ${#registered[@]} -gt 0 ]; then
  echo "Registered as '$NAME' in: ${registered[*]}. Restart the agent to load the tools."
  echo "Then ask it to research something, or in Claude Code type: /mcp__${NAME}__research <topic>"
else
  [ "$REGISTER" = 1 ] && warn "agent '$AGENT' not found on PATH"
  echo "Not registered anywhere. Add manually: <agent> mcp add $NAME -- uv --directory \"$DIR\" run research-mcp"
fi
[ "$REDLIB" = 1 ] || echo "Read Reddit threads: re-run with --redlib (or: cd \"$DIR\" && uv run kb redlib install)"
[ "$SEARXNG" = 1 ] || echo "Free keyless web search: re-run with --searxng (or: cd \"$DIR\" && uv run kb searxng install)"
echo "Optional free keys make search better (GitHub, Hugging Face, Semantic Scholar, Civitai, Tavily):"
echo "  cd \"$DIR\" && uv run kb keys            # what each key improves and where to get it"
echo "  uv run kb keys set TAVILY_API_KEY tvly-…   # stored in ~/.config/research-mcp/keys.env"
exit 0
