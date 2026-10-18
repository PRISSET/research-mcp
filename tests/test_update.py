"""Self-update: fast-forward only, never over local work — on throwaway git repos."""

import subprocess

from kb import update


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   capture_output=True)


def _repos(tmp_path):
    remote, clone = tmp_path / "remote", tmp_path / "clone"
    _git(tmp_path, "init", "-q", "-b", "main", str(remote))
    (remote / "a.txt").write_text("1")
    _git(remote, "add", ".")
    _git(remote, "commit", "-qm", "first")
    _git(tmp_path, "clone", "-q", str(remote), str(clone))
    (remote / "a.txt").write_text("2")
    _git(remote, "commit", "-qam", "Second: faster search")
    return remote, clone


def test_fast_forwards_a_clean_checkout(tmp_path, monkeypatch):
    _, clone = _repos(tmp_path)
    monkeypatch.setattr(update, "ROOT", clone)
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path / "data"))
    assert update.check(apply=False)["status"] == "update available"
    r = update.check()
    assert r["status"] == "updated" and r["changes"] == "Second: faster search"
    assert (clone / "a.txt").read_text() == "2" and update.check()["status"] == "up to date"


def test_never_touches_local_work(tmp_path, monkeypatch):
    _, clone = _repos(tmp_path)
    monkeypatch.setattr(update, "ROOT", clone)
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path / "data"))
    (clone / "a.txt").write_text("my edit")
    assert update.check()["status"].startswith("skipped: local changes")
    _git(clone, "commit", "-qam", "mine")
    assert update.check()["status"].startswith("skipped: local commits")
    assert (clone / "a.txt").read_text() == "my edit"


def test_moves_installs_off_the_rewritten_history(tmp_path, monkeypatch):
    remote, clone = _repos(tmp_path)
    old = subprocess.run(["git", "-C", str(clone), "rev-parse", "--short=7", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    _git(remote, "checkout", "-q", "--orphan", "fresh")
    _git(remote, "commit", "-qm", "squashed")
    _git(remote, "branch", "-M", "fresh", "main")
    monkeypatch.setattr(update, "ROOT", clone)
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path / "data"))
    assert update.check()["status"].startswith("skipped: local commits")      # unknown history: left alone
    monkeypatch.setattr(update, "REWRITTEN", {old})
    assert update.check()["status"] == "updated" and (clone / "a.txt").read_text() == "2"


def test_refresh_rules_only_where_installed(tmp_path, monkeypatch):
    from kb import rules, update

    monkeypatch.setenv("HOME", str(tmp_path))
    claude = tmp_path / ".claude" / "CLAUDE.md"
    claude.parent.mkdir()
    claude.write_text(f"mine\n\n{rules.START}\nold rules\n{rules.END}\n")
    assert update.refresh_rules() == [str(claude)]
    text = claude.read_text()
    assert text.startswith("mine") and "old rules" not in text and "study_topic" in text
    assert not (tmp_path / ".codex" / "AGENTS.md").exists()          # never added where the user did not install
