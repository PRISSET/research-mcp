import json
import subprocess

import pytest

from kb import codereview, guard


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "_path", lambda root: tmp_path / "ledgers" / "l.json")
    r = tmp_path / "proj"
    (r / "pkg").mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", "-C", str(r), *a], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (r / "pkg" / "util.py").write_text(
        "def load_items(path):\n    with open(path) as f:\n        data = f.read()\n    rows = data.split()\n"
        "    rows = [x.strip() for x in rows]\n    return [x for x in rows if x]\n\n\n"
        + "".join(f"def helper_{i}():\n    return {i}\n\n\n" for i in range(15)))
    git("add", ".")
    git("commit", "-qm", "init")
    return r


def test_ledger_items_close_with_where_or_why(repo):
    out = guard.update(str(repo), add=[{"kind": "optimization", "text": "read headers with one range request"},
                                       {"kind": "fact", "text": "the GPU is an RTX A6000"},
                                       {"kind": "optimization", "text": "read headers with one range request"}])
    assert out["open"] == ["#1 optimization: read headers with one range request"]
    assert out["standing"] == ["#2 fact: the GPU is an RTX A6000"]
    out = guard.update(str(repo), done=[{"id": 1}])
    assert "say where" in out["notes"][0] and out["open"]
    out = guard.update(str(repo), done=[{"id": 1, "where": "pkg/util.py:10"}])
    assert out["open"] == [] and out["standing"]


def test_hooks_show_ledger_and_stop_refuses_once(repo):
    guard.update(str(repo), add=[{"kind": "requirement", "text": "batch the requests"}])
    ctx = guard.hook("prompt", {"cwd": str(repo)})["hookSpecificOutput"]["additionalContext"]
    assert "OPEN #1 requirement: batch the requests" in ctx
    stop = guard.hook("stop", {"cwd": str(repo)})
    assert stop["decision"] == "block" and "batch the requests" in stop["reason"]
    assert guard.hook("stop", {"cwd": str(repo), "stop_hook_active": True}) is None   # no loop
    guard.update(str(repo), defer=[{"id": 1, "why": "the API has no batch endpoint"}])
    assert guard.hook("stop", {"cwd": str(repo)}) is None

    (repo / "pkg" / "new.py").write_text("def f():\n    return 1\n")
    stop = guard.hook("stop", {"cwd": str(repo)})
    assert "review_code" in stop["reason"]
    codereview.review_code(str(repo))
    assert guard.hook("stop", {"cwd": str(repo)}) is None                             # reviewed, unchanged
    (repo / "pkg" / "new.py").write_text("def f():\n    return 2\n")
    assert guard.hook("stop", {"cwd": str(repo)})["decision"] == "block"


def test_review_flags_what_the_change_added(repo, monkeypatch):
    monkeypatch.setattr(codereview, "_tool", lambda root, name: None)   # no linters: only our checks
    long_body = "".join(f"    x{i} = {i}\n" for i in range(70))
    (repo / "pkg" / "feature.py").write_text(
        "def read_all(path):\n    with open(path) as f:\n        text = f.read()\n    parts = text.split()\n"
        "    parts = [p.strip() for p in parts]\n    return [p for p in parts if p]\n\n\n"
        f"def big():\n{long_body}    return 1\n\n\n"
        "def risky():\n    try:\n        return 1 / 0\n    except Exception:\n        pass\n\n\n"
        "def show():\n    print('debug')  # TODO remove\n    s = '# TODO not a comment'\n    return s\n")
    r = codereview.review_code(str(repo))
    rules = {(f["rule"], f["line"]) for f in r["findings"]}
    assert r["changed_files"] == ["pkg/feature.py"]
    assert ("duplicate", 1) in rules                    # same body as load_items() in util.py
    assert ("long-function", 9) in rules
    assert any(x[0] == "swallowed-exception" for x in rules)
    assert any(x[0] == "near-duplicate" for x in rules) or ("duplicate", 1) in rules
    assert {"debug-output", "todo"} <= {x[0] for x in rules}
    assert sum(1 for x in rules if x[0] == "todo") == 1
    dup = next(f for f in r["findings"] if f["rule"] == "duplicate")
    assert "pkg/util.py:1" in dup["message"]


def test_old_code_is_not_judged(repo, monkeypatch):
    monkeypatch.setattr(codereview, "_tool", lambda root, name: None)
    p = repo / "pkg" / "util.py"
    p.write_text(p.read_text().replace("def helper_0():\n    return 0", "def helper_0():\n    return 100"))
    assert codereview.review_code(str(repo))["findings"] == []


def test_install_keeps_other_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    other = {"matcher": "Bash", "hooks": [{"type": "command", "command": "my-check.sh"}]}
    (tmp_path / "settings.json").write_text(json.dumps({"model": "opus", "hooks": {"PreToolUse": [other],
                                                                                   "Stop": [other]}}))
    guard.install()
    guard.install()   # idempotent
    cfg = json.loads((tmp_path / "settings.json").read_text())
    assert cfg["model"] == "opus" and cfg["hooks"]["PreToolUse"] == [other]
    assert len(cfg["hooks"]["Stop"]) == 2 and cfg["hooks"]["SessionStart"][0]["matcher"] == "compact|resume"
    assert guard.uninstall()
    cfg = json.loads((tmp_path / "settings.json").read_text())
    assert cfg["hooks"] == {"PreToolUse": [other], "Stop": [other]}


def test_hook_entry_point_never_breaks_the_session(monkeypatch, capsys):
    monkeypatch.setattr(guard, "hook", lambda e, d: 1 / 0)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{}"))
    assert guard.main(["stop"]) == 0
    assert capsys.readouterr().out == ""


def test_same_name_near_copy_is_flagged_but_namesakes_are_not(repo, monkeypatch):
    monkeypatch.setattr(codereview, "_tool", lambda root, name: None)
    (repo / "pkg" / "other.py").write_text(
        "def load_items(path):\n    with open(path) as f:\n        data = f.read()\n    rows = data.split()\n"
        "    rows = [x.strip() for x in rows]\n    rows = [x.lower() for x in rows]\n    return [x for x in rows if x]\n"
        "\n\ndef helper_1(a, b):\n    total = a + b\n    if total > 10:\n        return total * 2\n    return total\n")
    rules = [f["rule"] for f in codereview.review_code(str(repo))["findings"]]
    assert rules == ["near-duplicate"]      # helper_1 shares only the name


def test_nesting_counts_branches_not_elif_chains():
    import ast

    flat = "def f(x):\n" + "".join(f"    {'if' if i == 0 else 'elif'} x == {i}:\n        return {i}\n"
                                   for i in range(8))
    deep = "def g(x):\n    for a in x:\n        if a:\n            while a:\n                if a > 1:\n" \
           "                    with open(a) as f:\n                        if f:\n                            pass\n"
    assert codereview._depth(ast.parse(flat).body[0]) == 1
    assert codereview._depth(ast.parse(deep).body[0]) == 5
