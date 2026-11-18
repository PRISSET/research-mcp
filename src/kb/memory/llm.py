"""The local memory model: an uncensored (abliterated) Gemma 4 E4B in llama.cpp's llama-server.

It runs only while a finished chat is digested (started, used, stopped), so it holds no RAM or VRAM the rest of
the time. CPU by default: the video card belongs to ComfyUI; KB_MEMORY_GPU_LAYERS moves layers to a GPU build.
Abliterated weights do not refuse adult / NSFW chats, so those get remembered like any other.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tarfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

MODEL_REPO = "huihui-ai/Huihui-gemma-4-E4B-it-qat-q4_0-unquantized-abliterated-GGUF"
MODEL_FILE = "Huihui-gemma-4-E4B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf"
RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
CPU_ASSET = "-bin-ubuntu-x64.tar.gz"
START_WAIT = 180.0
MIN_FREE_GB = float(os.environ.get("KB_MEMORY_MIN_FREE_GB", "10"))   # model + context need ~6.5 GB
FREE_WAIT = float(os.environ.get("KB_MEMORY_FREE_WAIT", "3600"))     # how long to wait for that much free RAM


class Unavailable(Exception):
    pass


def home() -> Path:
    return Path(os.environ.get("KB_MEMORY_DIR") or "~/.local/share/research-mcp/memory").expanduser()


def model_path() -> Path:
    return Path(os.environ.get("KB_MEMORY_MODEL") or home() / "models" / "gemma-4-e4b-abliterated-Q4_K.gguf"
                ).expanduser()


def port() -> int:
    return int(os.environ.get("KB_MEMORY_PORT", "3087"))


def url() -> str:
    return f"http://127.0.0.1:{port()}"


def server_binary() -> Path | None:
    """KB_LLAMA_SERVER, then llama-server on PATH, then the copy `kb memory install` put in the memory folder."""
    named = os.environ.get("KB_LLAMA_SERVER") or shutil.which("llama-server")
    if named:
        return Path(named)
    found = sorted((home() / "llama").glob("**/llama-server"))
    return found[-1] if found else None


def status() -> dict[str, Any]:
    binary, model = server_binary(), model_path()
    return {"server": str(binary) if binary else None, "model": str(model), "model_ready": model.exists(),
            "running": _healthy(), "gpu_layers": int(os.environ.get("KB_MEMORY_GPU_LAYERS", "0"))}


def threads() -> int:
    """A third of the cores by default: the digest runs in the background and is not in a hurry."""
    return int(os.environ.get("KB_MEMORY_THREADS") or max(2, (os.cpu_count() or 6) // 3))


def free_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        pass
    return float("inf")   # not Linux: do not block


def background() -> None:
    """This process and what it starts get the CPU only when nothing else wants it (Linux SCHED_IDLE), and the
    disk last; other systems get the lowest nice."""
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, OSError):
        try:
            os.nice(19)
        except OSError:
            pass
    if shutil.which("ionice"):
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=False, capture_output=True)


def wait_for_memory(need: float = MIN_FREE_GB, wait: float = FREE_WAIT, step: float = 30.0) -> None:
    """Hold the start while RAM is short (ComfyUI running); give up after `wait` so the chat is tried next time."""
    deadline = time.monotonic() + wait
    while free_gb() < need:
        if time.monotonic() > deadline:
            raise Unavailable(f"only {free_gb():.1f} GB RAM free for {wait / 60:.0f} min (need {need:g}): "
                              "digest postponed to the next chat end")
        time.sleep(step)


def _healthy() -> bool:
    try:
        return httpx.get(f"{url()}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def _die_with_parent() -> None:
    """Linux: the server gets SIGTERM when the process that started it dies, so it never lingers holding RAM."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").prctl(1, 15)   # PR_SET_PDEATHSIG, SIGTERM
    except (OSError, AttributeError):
        pass


@contextmanager
def running() -> Iterator[None]:
    """A llama-server for the duration of the block; an already running one is used and left running."""
    if _healthy():
        yield
        return
    binary, model = server_binary(), model_path()
    if binary is None or not model.exists():
        raise Unavailable("the memory model is not installed: run `uv run kb memory install`")
    cmd = [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port()),
           "-c", os.environ.get("KB_MEMORY_CTX", "12288"), "-t", str(threads()), "-tb", str(threads()), "-np", "1",
           "-ngl", os.environ.get("KB_MEMORY_GPU_LAYERS", "0")]
    wait_for_memory()
    home().mkdir(parents=True, exist_ok=True)
    with open(home() / "llama-server.log", "ab") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                preexec_fn=_die_with_parent)
    try:
        deadline = time.monotonic() + START_WAIT
        while not _healthy():
            if proc.poll() is not None or time.monotonic() > deadline:
                raise Unavailable(f"llama-server did not start (see {home() / 'llama-server.log'})")
            time.sleep(1)
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def chat(messages: list[dict[str, str]], schema: dict[str, Any] | None = None, max_tokens: int = 2048,
         timeout: float = 1200.0) -> str:
    body: dict[str, Any] = {"messages": messages, "temperature": 0.2, "max_tokens": max_tokens,
                            "chat_template_kwargs": {"enable_thinking": False}}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    try:
        r = httpx.post(f"{url()}/v1/chat/completions", json=body, timeout=timeout)
    except httpx.HTTPError as e:
        raise Unavailable(f"memory model at {url()}: {type(e).__name__}: {e}") from e
    if r.status_code != 200:
        raise Unavailable(f"memory model answered {r.status_code}: {r.text[:300]}")
    try:
        return r.json()["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise Unavailable(f"unexpected answer from the memory model: {r.text[:300]}") from e


# ---------------- install ----------------

def _download(src: str, dest: Path) -> None:
    """Stream to a .part file and rename, so a broken download is never taken for a finished one."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with httpx.stream("GET", src, follow_redirects=True, timeout=httpx.Timeout(60, read=300)) as r:
        r.raise_for_status()
        with open(part, "wb") as fh:
            for block in r.iter_bytes(1 << 20):
                fh.write(block)
    part.replace(dest)


def install_server() -> Path:
    """The newest llama.cpp CPU build for Linux x64 from GitHub releases."""
    releases = httpx.get(RELEASES, timeout=30, follow_redirects=True).json()
    asset = next((a for rel in releases for a in rel.get("assets", []) if a["name"].endswith(CPU_ASSET)
                  and a["name"].startswith("llama-")), None)
    if asset is None:
        raise Unavailable("no llama.cpp Linux x64 build in the recent releases: install llama-server yourself "
                          "and set KB_LLAMA_SERVER")
    target = home() / "llama"
    archive = target / asset["name"]
    _download(asset["browser_download_url"], archive)
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    archive.unlink()
    binary = server_binary()
    if binary is None:
        raise Unavailable(f"llama-server not found in {asset['name']}")
    return binary


def install() -> dict[str, str]:
    out = {}
    if not model_path().exists():
        _download(f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}", model_path())
        out["model"] = "downloaded"
    out["server"] = str(server_binary() or install_server())
    return out
