import json

from kb import opencode

ENV = {"RESEARCH_NOTEBOOK": "/n/RESEARCH.md", "RESEARCH_LANG": "ru", "CUDA_VISIBLE_DEVICES": ""}


def _setup(tmp_path, monkeypatch, version, cfg=None):
    path = tmp_path / "opencode" / "opencode.json"
    path.parent.mkdir()
    if cfg is not None:
        path.write_text(cfg if isinstance(cfg, str) else json.dumps(cfg))
    monkeypatch.setenv("OPENCODE_CONFIG", str(path))
    monkeypatch.setattr(opencode, "major_version", lambda: version)
    return path


def test_v2_merges_under_servers_and_keeps_the_rest(tmp_path, monkeypatch):
    path = _setup(tmp_path, monkeypatch, 2, {"model": "llama.cpp/qwen", "provider": {"llama.cpp": {}},
                                             "mcp": {"research": {"type": "local"}, "servers": {"other": {}}}})
    opencode.register("research", "/repo", ENV)
    cfg = json.loads(path.read_text())
    assert cfg["model"] == "llama.cpp/qwen" and "llama.cpp" in cfg["provider"]
    srv = cfg["mcp"]["servers"]["research"]
    assert srv["command"] == ["uv", "--directory", "/repo", "run", "research-mcp"]
    assert srv["environment"] == ENV and srv["timeout"] == {"startup": 120_000}
    assert "other" in cfg["mcp"]["servers"] and "research" not in cfg["mcp"]   # old v1-style entry moved
    assert path.with_suffix(".json.bak").exists()
    assert opencode.unregister("research") and "research" not in json.loads(path.read_text())["mcp"]["servers"]


def test_v1_format_and_new_file(tmp_path, monkeypatch):
    path = _setup(tmp_path, monkeypatch, 1)
    opencode.register("research", "/repo", ENV)
    srv = json.loads(path.read_text())["mcp"]["research"]
    assert srv["enabled"] is True and srv["timeout"] == 120_000 and srv["type"] == "local"


def test_jsonc_goes_through_the_opencode_cli(tmp_path, monkeypatch):
    path = _setup(tmp_path, monkeypatch, 2, '{\n  // my comment\n  "model": "x",\n}\n')
    calls = []
    monkeypatch.setattr(opencode.shutil, "which", lambda name: "/bin/opencode")
    monkeypatch.setattr(opencode.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    opencode.register("research", "/repo", ENV)
    assert calls[0][:5] == ["opencode", "mcp", "add", "research", "--global"]
    assert "RESEARCH_LANG=ru" in calls[0] and calls[0][-4:] == ["--directory", "/repo", "run", "research-mcp"]
    assert "// my comment" in path.read_text()   # the user's file is not rewritten by us
