import json

import pytest

from kb import nudge


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_MEMORY_DIR", str(tmp_path / "rm" / "memory"))
    monkeypatch.setattr(nudge, "_proxy_up", lambda: True)


def _call(tool, **args):
    return {"tool_name": tool, "tool_input": args, "session_id": "s1", "cwd": "/proj"}


def test_first_heavy_read_is_sent_to_delegate_then_allowed(tmp_path):
    big = tmp_path / "big.py"
    big.write_text("x" * 50_000)
    out = nudge.hook(_call("Read", file_path=str(big)))
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny" and 'project="/proj"' in reason
    assert nudge.hook(_call("Read", file_path=str(big))) is None                 # the same call again goes through
    assert nudge.hook(_call("WebFetch", url="https://a.example/doc"))["hookSpecificOutput"]
    assert nudge.hook({**_call("WebFetch", url="https://a.example/doc"), "session_id": "s2"})   # per session


@pytest.mark.parametrize("call", [
    _call("Read", file_path="/etc/hostname"),
    _call("mcp__research__fetch", url="https://a.example"),
    _call("mcp__research__fetch", urls=["https://a.example"]),
    _call("Bash", command="cat big"),
])
def test_light_calls_pass(call):
    assert nudge.hook(call) is None


def test_ranges_images_and_several_pages(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 50_000)
    assert nudge.target("Read", {"file_path": str(big), "offset": 100, "limit": 50}) is None
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"x" * 50_000)
    assert nudge.target("Read", {"file_path": str(shot)}) is None
    assert nudge.target("mcp__research__fetch", {"urls": ["b", "a"]}) == "a b"


def test_nothing_is_refused_while_the_proxy_is_down(tmp_path, monkeypatch):
    monkeypatch.setattr(nudge, "_proxy_up", lambda: False)
    assert nudge.hook(_call("WebFetch", url="https://a.example")) is None


def test_install_uses_the_tool_matcher(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    nudge.install()
    nudge.install()
    (group,) = json.loads((tmp_path / "settings.json").read_text())["hooks"]["PreToolUse"]
    assert group["matcher"] == nudge.MATCHER and "kb.nudge" in group["hooks"][0]["command"]
    assert nudge.uninstall()


def test_a_broken_hook_never_blocks(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", type("S", (), {"read": staticmethod(lambda: "{not json")})())
    assert nudge.main() == 0 and capsys.readouterr().out == ""
