import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from kb import deepseek, delegate


@pytest.fixture
def project(tmp_path):
    r = tmp_path / "proj"
    r.mkdir()
    (r / "train.py").write_text("LR = 1e-4\n\ndef train():\n    return LR\n")
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    subprocess.run(["git", "-C", str(r), "add", "."], check=True)
    return r


def _scripted(monkeypatch, replies):
    sent = []

    def chat(messages, tools=None, timeout=240.0):
        sent.append({"messages": [dict(m) for m in messages], "tools": tools})
        return replies.pop(0)

    monkeypatch.setattr(deepseek, "ensure_running", lambda: None)
    monkeypatch.setattr(deepseek, "chat", chat)
    return sent


def _call(name, **args):
    return {"id": f"c-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_subagent_reads_the_project_and_returns_only_the_answer(monkeypatch, project):
    real = delegate.run_tool
    monkeypatch.setattr(delegate, "run_tool", lambda name, args, root: (
        "see https://github.com/kohya-ss/sd-scripts/issues/1492" if name == "web_search" else real(name, args, root)))
    sent = _scripted(monkeypatch, [
        {"content": "", "tool_calls": [_call("grep", pattern="LR ="), _call("web_search", query="lr")]},
        {"content": "", "tool_calls": [_call("read_file", path="train.py", start=1, lines=2)]},
        {"content": "LR = 1e-4 (train.py:1), per https://github.com/kohya-ss/sd-scripts/issues/1492; "
                    "see also https://example.com/made-up"},
    ])
    out = delegate.delegate("what lr does train.py use?", project=str(project))
    assert out["answer"].startswith("LR = 1e-4") and out["steps"] == 3
    assert out["sources"] == ["https://github.com/kohya-ss/sd-scripts/issues/1492"]
    assert out["unverified_links"] == ["https://example.com/made-up"]
    tool_msgs = [m for m in sent[2]["messages"] if m["role"] == "tool"]
    assert "train.py:1:LR = 1e-4" in tool_msgs[0]["content"] and "1: LR = 1e-4" in tool_msgs[2]["content"]
    names = {t["function"]["name"] for t in sent[0]["tools"]}
    assert {"fetch", "search_fulltext", "web_search", "read_file", "grep"} <= names
    assert "notebook_write" not in names                     # read-only


def test_last_step_asks_for_the_answer_without_tools(monkeypatch, project):
    sent = _scripted(monkeypatch, [{"content": "", "tool_calls": [_call("grep", pattern="x")]},
                                   {"content": "nothing found"}])
    out = delegate.delegate("q", project=str(project), max_steps=2)
    assert out["answer"] == "nothing found" and sent[1]["tools"] is None
    assert "final answer" in sent[1]["messages"][-1]["content"]


def test_files_outside_the_project_are_refused(project):
    assert "outside the project" in delegate.run_tool("read_file", {"path": "../../etc/passwd"}, project)
    assert "unknown tool" in delegate.run_tool("read_file", {"path": "train.py"}, None)


def test_old_tool_results_shrink_first(monkeypatch):
    monkeypatch.setattr(delegate, "CONTEXT_CHARS", 4500)
    msgs = [{"role": "user", "content": "task"}] + [{"role": "tool", "content": "x" * 2000} for _ in range(3)]
    delegate._fit(msgs)
    assert len(msgs[1]["content"]) < 800 and len(msgs[2]["content"]) < 800 and len(msgs[3]["content"]) == 2000


def test_no_token_means_do_it_yourself(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_CHAT_TOKEN", raising=False)
    out = delegate.delegate("q")
    assert "DEEPSEEK_CHAT_TOKEN" in out["error"] and "yourself" in out["next"]


def test_token_from_local_storage_json(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_CHAT_TOKEN", '{"value":"abc","__version":"0"}')
    assert deepseek.token() == "abc"
    monkeypatch.setenv("DEEPSEEK_CHAT_TOKEN", "t1|t2")
    assert deepseek.token() == "t1|t2"


def test_chat_talks_to_the_proxy(monkeypatch):
    got = {}

    class Proxy(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def do_POST(self):
            got["auth"] = self.headers["Authorization"]
            got["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "hi"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Proxy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("DEEPSEEK_PROXY_URL", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("DEEPSEEK_CHAT_TOKEN", "tok")
    try:
        assert deepseek.chat([{"role": "user", "content": "q"}])["content"] == "hi"
    finally:
        srv.shutdown()
    assert got["auth"] == "Bearer tok" and got["body"]["client_prompt"] is True and got["body"]["stream"] is False


def test_proxy_never_holds_the_token(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_CHAT_TOKEN", "secret")
    env = deepseek._proxy_env()
    assert "DEEPSEEK_CHAT_TOKEN" not in env and env["PROXY_PORT"] == str(deepseek.PORT)
