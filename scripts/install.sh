#!/usr/bin/env bash
# One-command install of the research MCP server and its registration in Claude Code and Codex CLI.
#
#   curl -fsSL https://raw.githubusercontent.com/PRISSET/research-mcp/main/scripts/install.sh | bash
#   scripts/install.sh [--dir PATH] [--notebook FILE] [--lang ru|en] [--embed|--no-embed] [--agent-rules] [--no-register]
#
#   --dir PATH       where to clone (default: this checkout, or ~/research-mcp)
#   --notebook FILE  the shared research notebook (default: ~/research/RESEARCH.md)
#   --lang ru|en     language of the notebook headings (default: ru)
#   --embed          semantic search: a static embedding model (model2vec, no torch, ~130 MB RAM) (default)
#   --no-embed       skip it: keyword search only
# The server runs on the CPU (registered with CUDA_VISIBLE_DEVICES=""): the GPU stays free for your own work.
#   --agent-rules    add a short rules block to ~/.claude/CLAUDE.md and ~/.codex/AGENTS.md so agents in every project
#                    look models / versions / errors up with this MCP (marked block; remove: --remove-agent-rules)
#   --no-register    do not touch Claude Code / Codex CLI configs
# Safe to re-run: pulls updates, re-syncs, re-registers. No API keys needed.
set -euo pipefail

REPO_URL="https://github.com/PRISSET/research-mcp.git"
NAME="research"
DIR="" NOTEBOOK="$HOME/research/RESEARCH.md" LANG_NB="ru" EMBED=1 REGISTER=1 RULES=keep

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift ;;
    --notebook) NOTEBOOK="$2"; shift ;;
    --lang) LANG_NB="$2"; shift ;;
    --embed) EMBED=1 ;;
    --no-embed) EMBED=0 ;;
    --no-register) REGISTER=0 ;;
    --agent-rules) RULES=install ;;
    --remove-agent-rules) RULES=remove ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*" >&2; }

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
  if command -v claude >/dev/null; then
    say "Registering in Claude Code (user scope)"
    for old in "$NAME" h3-kb; do claude mcp remove "$old" -s user >/dev/null 2>&1 || true; done
    claude mcp add "$NAME" -s user -e "${envs[0]}" -e "${envs[1]}" -e "${envs[2]}" -- uv --directory "$DIR" run research-mcp \
      && registered+=("Claude Code")
  fi
  if command -v codex >/dev/null; then
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
fi

# --- optional: global agent rules (Claude Code reads ~/.claude/CLAUDE.md, Codex ~/.codex/AGENTS.md)
if [ "$RULES" != keep ]; then
  say "Agent rules: $RULES"
  targets=()
  command -v claude >/dev/null && targets+=("$HOME/.claude/CLAUDE.md")
  command -v codex >/dev/null && targets+=("${CODEX_HOME:-$HOME/.codex}/AGENTS.md")
  if [ ${#targets[@]} -gt 0 ]; then
    uv run python -m kb.rules "$RULES" "${targets[@]}"
  else
    warn "neither claude nor codex found: no rules written"
  fi
fi

say "Done"
echo "Installed in: $DIR"
echo "Notebook:     $NOTEBOOK (created on the first research)"
if [ ${#registered[@]} -gt 0 ]; then
  echo "Registered as '$NAME' in: ${registered[*]}. Restart the agent to load the tools."
  echo "Then ask it to research something, or in Claude Code type: /mcp__${NAME}__research <topic>"
else
  echo "Not registered anywhere. Add manually: <agent> mcp add $NAME -- uv --directory \"$DIR\" run research-mcp"
fi
exit 0
