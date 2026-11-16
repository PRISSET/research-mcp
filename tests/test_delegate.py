import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import pytest

from kb import delegate, gemini


@pytest.fixture(autouse=True)
def no_pacing(monkeypatch, tmp_path):
    monkeypatch.setattr(gemini, "MIN_GAP", 0.0)
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path / "data"))


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

    def chat(messages, timeout=240.0):
        sent.append([dict(m) for m in messages])
        return replies.pop(0)

    monkeypatch.setattr(gemini, "chat", chat)
    return sent


def _call(name, **args):
    return "```json\n" + json.dumps({"name": name, "arguments": args}) + "\n```"


def test_subagent_reads_the_project_and_returns_only_the_answer(monkeypatch, project):
    real = delegate.run_tool
    monkeypatch.setattr(delegate, "run_tool", lambda name, args, root: (
        "see https://github.com/kohya-ss/sd-scripts/issues/1492 id gh:kohya-ss/sd-scripts#1492"
        if name == "web_search" else real(name, args, root)))
    answer = ("LR = 1e-4 (train.py:1), per https://github.com/kohya-ss/sd-scripts/issues/1492 "
              "([gh:kohya-ss/sd-scripts#1492](https://github.com/kohya-ss/sd-scripts/issues/1492)); "
              "see also https://example.com/made-up, web:example.com/x")
    sent = _scripted(monkeypatch, [_call("grep", pattern="LR =") + "\n" + _call("web_search", query="lr"),
                                   _call("read_file", path="train.py", start=1, lines=2), answer])
    out = delegate.delegate("what lr does train.py use?", project=str(project))
    assert out["answer"].startswith("LR = 1e-4") and out["steps"] == 3
    assert out["sources"] == ["https://github.com/kohya-ss/sd-scripts/issues/1492", "gh:kohya-ss/sd-scripts#1492"]
    assert out["unverified_links"] == ["https://example.com/made-up", "web:example.com/x"]
    outputs = [m["content"] for m in sent[2] if m.get("role") == "user" and m["content"].startswith("Outputs:")]
    assert "train.py:1:LR = 1e-4" in outputs[0] and "1: LR = 1e-4" in outputs[1]
    system = sent[0][0]["content"]
    assert all(f"- {n}(" in system for n in ("fetch", "search_fulltext", "web_search", "read_file", "grep"))
    assert "- notebook_write(" not in system                     # read-only
    assert all(set(m) == {"role", "content"} for m in sent[2])    # only what the proxy understands


def test_parse_calls_takes_blocks_and_lists():
    text = (_call("fetch", url="https://a") + "\nsome words\n```json\n[{\"name\": \"grep\", \"arguments\": "
            "{\"pattern\": \"x\"}}, {\"name\": \"read_file\"}]\n```\n```json\n{not json}\n```")
    assert delegate.parse_calls(text) == [{"name": "fetch", "arguments": {"url": "https://a"}},
                                          {"name": "grep", "arguments": {"pattern": "x"}},
                                          {"name": "read_file", "arguments": {}}]


def test_last_step_asks_for_the_answer_without_calls(monkeypatch, project):
    sent = _scripted(monkeypatch, [_call("grep", pattern="x"), "nothing found"])
    out = delegate.delegate("q", project=str(project), max_steps=2)
    assert out["answer"] == "nothing found"
    assert "final answer" in sent[1][-1]["content"]


def test_unknown_command_is_retried_not_run(monkeypatch, project):
    sent = _scripted(monkeypatch, [_call("bash", command="echo check"), _call("grep", pattern="LR"),
                                   "LR = 1e-4 (train.py:1)"])
    out = delegate.delegate("q", project=str(project))
    assert out["answer"] == "LR = 1e-4 (train.py:1)" and out["tool_calls"] == ['grep({"pattern": "LR"})']
    nudge = sent[1][-1]["content"]
    assert "not run" in nudge and "grep" in nudge and "fetch" in nudge


def test_endless_unknown_commands_end_in_an_error(monkeypatch, project):
    _scripted(monkeypatch, [_call("bash", command="ls")] * 5)
    out = delegate.delegate("q", project=str(project))
    assert "cannot run" in out["error"] and "answer" not in out


def test_files_outside_the_project_are_refused(project):
    assert "outside the project" in delegate.run_tool("read_file", {"path": "../../etc/passwd"}, project)
    assert "unknown tool" in delegate.run_tool("read_file", {"path": "train.py"}, None)


def test_old_results_shrink_first(monkeypatch):
    monkeypatch.setattr(delegate, "CONTEXT_CHARS", 4500)
    msgs = [{"role": "user", "content": "task"}] + [{"role": "user", "content": "x" * 2000, "result": True}
                                                    for _ in range(3)]
    delegate._fit(msgs)
    assert len(msgs[1]["content"]) < 800 and len(msgs[2]["content"]) < 800 and len(msgs[3]["content"]) == 2000


def test_markdown_images_do_not_reach_the_model(project):
    (project / "README.md").write_text("# X\n![overview](assets/overview.png) text\n")
    out = delegate.run_tool("read_file", {"path": "README.md"}, project)
    assert "[image: overview]" in out and "assets/overview.png" not in out


def test_proxy_down_means_do_it_yourself(monkeypatch):
    monkeypatch.setenv("GEMINI_PROXY_URL", "http://127.0.0.1:9")
    out = delegate.delegate("q")
    assert "Gemini proxy" in out["error"] and "yourself" in out["next"]


def _fake_proxy(statuses, delay=0.0, threaded=False):
    """A proxy answering POSTs with the given statuses in turn (then 200); records what it got."""
    got = {"statuses": [], "now": 0, "peak": 0}
    lock = threading.Lock()

    class Proxy(BaseHTTPRequestHandler):
        def do_POST(self):
            got["auth"] = self.headers["Authorization"]
            got["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                got["now"] += 1
                got["peak"] = max(got["peak"], got["now"])
            time.sleep(delay)
            with lock:
                got["now"] -= 1
            status = statuses.pop(0) if statuses else 200
            got["statuses"].append(status)
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]} if status == 200
                              else {"error": {"message": "too many requests"}})
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            pass

    srv = (ThreadingHTTPServer if threaded else HTTPServer)(("127.0.0.1", 0), Proxy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, got


@pytest.fixture
def proxy(monkeypatch):
    servers = []

    def start(statuses=(), **kw):
        srv, got = _fake_proxy(list(statuses), **kw)
        servers.append(srv)
        monkeypatch.setenv("GEMINI_PROXY_URL", f"http://127.0.0.1:{srv.server_port}")
        return got

    yield start
    for srv in servers:
        srv.shutdown()


def test_chat_talks_to_the_proxy(monkeypatch, proxy):
    monkeypatch.setenv("GEMINI_PROXY_TOKEN", "pro")
    got = proxy()
    assert gemini.chat([{"role": "user", "content": "q"}]) == "ok"
    assert got["auth"] == "Bearer pro" and got["body"]["model"] == gemini.model()


@pytest.mark.parametrize("refusals, ok", [(2, True), (3, False)])
def test_refusals_wait_and_retry(monkeypatch, proxy, refusals, ok):
    got = proxy([429] * refusals)
    monkeypatch.setattr(gemini, "RATE_WAITS", (0.0, 0.0))     # two retries
    if ok:
        assert gemini.chat([{"role": "user", "content": "q"}]) == "ok"
    else:
        with pytest.raises(gemini.Unavailable, match="429"):
            gemini.chat([{"role": "user", "content": "q"}])
    assert got["statuses"] == ([429, 429, 200] if ok else [429, 429, 429])


def test_completions_keep_a_gap(monkeypatch, proxy):
    proxy()
    monkeypatch.setattr(gemini, "MIN_GAP", 0.4)
    t = time.monotonic()
    for _ in range(3):
        gemini.chat([{"role": "user", "content": "q"}])
    assert time.monotonic() - t >= 0.8


def test_parallel_chats_wait_for_each_other(proxy):
    got = proxy(delay=0.3, threaded=True)
    answers = []
    workers = [threading.Thread(target=lambda: answers.append(gemini.chat([{"role": "user", "content": "q"}])))
               for _ in range(3)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert answers == ["ok"] * 3 and got["peak"] == 1      # the next request went only after the previous finished


def test_calls_on_the_last_step_are_not_an_answer(monkeypatch, project):
    _scripted(monkeypatch, [_call("grep", pattern="x"), _call("grep", pattern="y")])
    out = delegate.delegate("q", project=str(project), max_steps=2)
    assert "out of steps" in out["error"] and "answer" not in out


def test_a_source_counts_only_when_read_exactly():
    report = delegate._report("see https://example.org/doc", {"https://example.org/docs/other"}, [], 0, 1)
    assert report["unverified_links"] == ["https://example.org/doc"]
