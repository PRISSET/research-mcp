"""review_code: check the agent's uncommitted changes before it says "done".

Only what the change added is judged, not the code it touched: the project's own linters (ruff, eslint — with the
project's config) on the changed lines; for Python, the functions the change wrote or grew: length, nesting,
parameters, swallowed exceptions, debug prints, TODOs, more comments than the surrounding code, and helpers that
already exist elsewhere in the project (the same body, or a same-named near copy) and should be reused. Then the task ledger:
what the user asked for that is still open.
"""

from __future__ import annotations

import ast
import io
import json
import re
import shutil
import subprocess
import tokenize
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from kb import guard

MAX_FUNC_LINES = 60
MAX_DEPTH = 4
MAX_PARAMS = 7
MAX_FINDINGS = 60
NEAR_COPY = 0.85      # body similarity of a same-named function that counts as a copy
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", ".tox", ".mypy_cache",
             "site-packages", ".pytest_cache"}


def added_lines(root: Path, base: str, files: list[str]) -> dict[str, set[int]]:
    """file -> line numbers (new side) the change added; untracked files count as wholly added."""
    out: dict[str, set[int]] = {f: set() for f in files}
    diff = guard._git(root, "diff", "-U0", "--no-color", base, "--", *files) or ""
    cur = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            cur = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@") and cur in out:
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            start, n = int(m.group(1)), int(m.group(2) or 1)
            out[cur].update(range(start, start + n))
    tracked = set((guard._git(root, "ls-files", "--", *files) or "").splitlines())
    for f in files:
        if f not in tracked:
            try:
                out[f] = set(range(1, len((root / f).read_text(encoding="utf-8", errors="replace").splitlines()) + 1))
            except OSError:
                pass
    return out


def _tool(root: Path, name: str) -> list[str] | None:
    for local in (root / ".venv" / "bin" / name, root / "node_modules" / ".bin" / name):
        if local.exists():
            return [str(local)]
    return [shutil.which(name)] if shutil.which(name) else None


def lint(root: Path, files: list[str], added: dict[str, set[int]]) -> tuple[list[dict[str, Any]], list[str]]:
    findings: list[dict[str, Any]] = []
    ran = []
    py = [f for f in files if f.endswith((".py", ".pyi"))]
    ruff = _tool(root, "ruff") if py else None
    if ruff:
        try:
            r = subprocess.run([*ruff, "check", "--output-format", "json", "--no-fix", *py], cwd=root,
                               capture_output=True, text=True, timeout=120)
            ran.append("ruff")
            for d in json.loads(r.stdout or "[]"):
                f = str(Path(d["filename"]).resolve().relative_to(root))
                line = (d.get("location") or {}).get("row")
                if line in added.get(f, set()):
                    findings.append({"file": f, "line": line, "rule": f"ruff {d.get('code')}",
                                     "message": d.get("message")})
        except (OSError, ValueError, subprocess.TimeoutExpired) as e:
            ran.append(f"ruff failed: {e}")
    js = [f for f in files if f.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte"))]
    eslint = _tool(root, "eslint") if js else None
    if eslint and (root / "node_modules").exists():
        try:
            r = subprocess.run([*eslint, "-f", "json", *js], cwd=root, capture_output=True, text=True, timeout=180)
            ran.append("eslint")
            for res in json.loads(r.stdout or "[]"):
                f = str(Path(res["filePath"]).resolve().relative_to(root))
                for m in res.get("messages") or []:
                    if m.get("line") in added.get(f, set()):
                        findings.append({"file": f, "line": m["line"], "rule": f"eslint {m.get('ruleId')}",
                                         "message": m.get("message")})
        except (OSError, ValueError, subprocess.TimeoutExpired) as e:
            ran.append(f"eslint failed: {e}")
    return findings, ran


# ---------------- Python structure ----------------

NEST = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Match)   # try / with do not branch
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _depth(node: ast.AST, d: int = 0) -> int:
    """Deepest nesting of branches and loops; an elif is the same level as its if."""
    best = d
    for field, value in ast.iter_fields(node):
        if isinstance(node, ast.If) and field == "orelse" and len(value) == 1 and isinstance(value[0], ast.If):
            best = max(best, _depth(value[0], d))
            continue
        for child in value if isinstance(value, list) else [value]:
            if isinstance(child, ast.AST) and not isinstance(child, SCOPES):
                best = max(best, _depth(child, d + 1) if isinstance(child, NEST) else _depth(child, d))
    return best


def _shape(fn: ast.AST) -> str:
    """The body with names, constants and docstrings blanked: equal shapes are copies of one helper."""
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant))]
    text = ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=False, include_attributes=False)
    return re.sub(r"'[^']*'|\b\d+\b", "_", text)


def _functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def project_index(root: Path, skip: set[str]) -> tuple[dict[str, list[tuple[str, str]]], dict[str, str]]:
    """Function name -> [(where, body shape)], and body shape -> where, over the project's other Python files
    (tests excluded: their helpers are not for reuse)."""
    names: dict[str, list[tuple[str, str]]] = {}
    shapes: dict[str, str] = {}
    files = [p for p in root.rglob("*.py") if not (set(p.relative_to(root).parts) & (SKIP_DIRS | {"tests", "test"}))
             and not p.name.startswith("test_")][:4000]
    for p in files:
        rel = str(p.relative_to(root))
        if rel in skip:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError, OSError):
            continue
        for fn in _functions(tree):
            where, shape = f"{rel}:{fn.lineno}", _shape(fn)
            names.setdefault(fn.name, []).append((where, shape))
            if (fn.end_lineno or fn.lineno) - fn.lineno >= 4:
                shapes.setdefault(shape, where)
    return names, shapes


def _comments(src: str) -> list[tuple[int, str]]:
    """(line, text) of real comments: a "# TODO" inside a string is not one."""
    try:
        return [(t.start[0], t.string) for t in tokenize.generate_tokens(io.StringIO(src).readline)
                if t.type == tokenize.COMMENT]
    except (tokenize.TokenError, SyntaxError):
        return []


def _comment_share(lines: list[str]) -> float:
    code = [ln for ln in lines if ln.strip()]
    return sum(1 for ln in code if ln.strip().startswith("#")) / len(code) if code else 0.0


def _hit(f: str, line: int | None, rule: str, message: str) -> dict[str, Any]:
    return {"file": f, "line": line, "rule": rule, "message": message}


def _function_checks(f: str, fn: ast.FunctionDef | ast.AsyncFunctionDef, new: set[int], index: Any,
                     is_test: bool) -> list[dict[str, Any]]:
    """Size, nesting, parameters and copies of one function the change wrote or edited."""
    names, shapes = index
    span = set(range(fn.lineno, (fn.end_lineno or fn.lineno) + 1))
    mine = span & new
    if not mine:
        return []
    out = []
    written = fn.lineno in new   # a new function, not an edit inside an old one
    where, n = f"{fn.name}()", len(span)
    if n > MAX_FUNC_LINES and (written or len(mine) > 10):
        out.append(_hit(f, fn.lineno, "long-function", f"{where} is {n} lines: split it into named steps"))
    if (d := _depth(fn)) > MAX_DEPTH:
        out.append(_hit(f, fn.lineno, "deep-nesting", f"{where} nests {d} levels: return early or extract the "
                                                       "inner block"))
    params = len(fn.args.args) + len(fn.args.kwonlyargs)
    if params > MAX_PARAMS and written:
        out.append(_hit(f, fn.lineno, "many-parameters", f"{where} takes {params} parameters"))
    if not written or is_test or n < 5:
        return out
    shape = _shape(fn)
    if dup := shapes.get(shape):
        out.append(_hit(f, fn.lineno, "duplicate", f"{where} repeats {dup}: reuse it"))
    elif near := [w for w, other in names.get(fn.name, []) if SequenceMatcher(None, other, shape).ratio() >= NEAR_COPY]:
        out.append(_hit(f, fn.lineno, "near-duplicate", f"{where} is nearly the same as {', '.join(near[:3])}: "
                                                         "reuse it or share the common part"))
    return out


def _silent_broad_except(node: ast.ExceptHandler) -> bool:
    broad = node.type is None or (isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"))
    return broad and all(isinstance(s, (ast.Pass, ast.Continue)) or
                         (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)) for s in node.body)


def _node_checks(f: str, tree: ast.AST, new: set[int], debug_ok: bool) -> list[dict[str, Any]]:
    """Swallowed exceptions and debug output on the added lines."""
    out = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if line not in new:
            continue
        if isinstance(node, ast.ExceptHandler) and _silent_broad_except(node):
            out.append(_hit(f, line, "swallowed-exception", "a broad except that does nothing hides failures: "
                                                            "catch the expected error, or log it"))
        elif (not debug_ok and isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in ("print", "breakpoint") and not any(k.arg == "file" for k in node.keywords)):
            out.append(_hit(f, line, "debug-output", f"{node.func.id}() left in library code: remove it or use "
                                                     "logging"))
    return out


def _text_checks(f: str, src: str, new: set[int]) -> list[dict[str, Any]]:
    """TODOs in real comments, and comment density against the rest of the file."""
    out = [_hit(f, ln, "todo", "TODO added: do it now or tell the user it is left") for ln, c in _comments(src)
           if ln in new and re.search(r"#\s*(TODO|FIXME|XXX|HACK)\b", c)]
    lines = src.splitlines()
    old = [lines[i - 1] for i in range(1, len(lines) + 1) if i not in new]
    mine = [lines[i - 1] for i in sorted(new) if i <= len(lines)]
    share, base = _comment_share(mine), _comment_share(old)
    if len(mine) >= 15 and len(old) >= 30 and share > max(2.5 * base, base + 0.15):
        out.append(_hit(f, min(new), "comment-density", f"the change is {share:.0%} comments, the file {base:.0%}: "
                                                         "match the surrounding code, keep only comments that say why"))
    return out


def python_checks(root: Path, files: list[str], added: dict[str, set[int]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    py = [f for f in files if f.endswith(".py") and added.get(f)]
    if not py:
        return findings
    index = project_index(root, set(py))
    for f in py:
        src = (root / f).read_text(encoding="utf-8", errors="replace")
        new = added[f]
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            findings.append(_hit(f, e.lineno, "syntax", str(e.msg)))
            continue
        is_test = "/tests/" in f"/{f}" or Path(f).name.startswith("test_")
        entry = bool(re.search(r"(^|/)(cli|scripts?|__main__)", f)) or "__name__ == \"__main__\"" in src
        for fn in _functions(tree):
            findings += _function_checks(f, fn, new, index, is_test)
        findings += _node_checks(f, tree, new, is_test or entry)
        findings += _text_checks(f, src, new)
    return findings


def generic_checks(root: Path, files: list[str], added: dict[str, set[int]]) -> list[dict[str, Any]]:
    findings = []
    for f in files:
        if f.endswith(".py") or not added.get(f):
            continue
        try:
            lines = (root / f).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for ln in sorted(added[f]):
            text = lines[ln - 1] if ln <= len(lines) else ""
            if re.search(r"(//|#|/\*)\s*(TODO|FIXME|XXX|HACK)\b", text):
                findings.append({"file": f, "line": ln, "rule": "todo",
                                 "message": "TODO added: do it now or tell the user it is left"})
            elif re.search(r"\bconsole\.(log|debug)\(|\bdebugger;|\bdbg!\(|fmt\.Println\(\"DEBUG", text):
                findings.append({"file": f, "line": ln, "rule": "debug-output", "message": "debug output left in"})
    return findings


def review_code(project: str | None = None, base: str | None = None, files: list[str] | None = None) -> dict[str, Any]:
    root = guard.project_root(project)
    if guard._git(root, "rev-parse", "--git-dir") is None:
        return {"project": str(root), "error": "not a git repository: review_code compares against git",
                "next": "pass project= the repository path (your working directory)"}
    base = base or guard._base(root)
    changed = guard.changed_code(root, base)
    if files:
        want = {str(Path(f)) for f in files}
        changed = [f for f in changed if f in want or any(f.startswith(w.rstrip("/") + "/") for w in want)]
    led = guard.load(root)
    out: dict[str, Any] = {"project": str(root), "base": base, "changed_files": changed}
    if changed:
        added = added_lines(root, base, changed)
        findings, ran = lint(root, changed, added)
        findings += python_checks(root, changed, added)
        findings += generic_checks(root, changed, added)
        findings.sort(key=lambda x: (x["file"], x["line"] or 0))
        out.update(linters=ran or ["none found in the project (ruff / eslint)"], findings=findings[:MAX_FINDINGS],
                   more=max(0, len(findings) - MAX_FINDINGS), added_lines=sum(len(v) for v in added.values()))
    out["ledger_open"] = [guard._line(i) for i in guard.open_items(led)]
    out["ledger_standing"] = [guard._line(i) for i in guard.standing(led)]
    if changed and not files:
        guard.mark_reviewed(root)
    out["next"] = ("Fix the findings that apply (structure, reuse, failures) or say why one does not; check the "
                   "change against ledger_standing (constraints, facts, decisions); finish the open ledger items "
                   "or mark them done (where) / deferred (why); in the final answer list which items were done "
                   "and where.")
    return out
