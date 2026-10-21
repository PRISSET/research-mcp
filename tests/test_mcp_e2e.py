"""End-to-end: a real MCP client (stdio) lists the tools and the /research prompt and fills RESEARCH.md."""

import json
import os
import sys

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from kb.schemas import Paper
from kb.store import Database

pytestmark = pytest.mark.slow


def _json(res) -> dict:
    return json.loads("".join(getattr(c, "text", "") for c in res.content))


async def _run(env: dict[str, str]) -> dict:
    params = StdioServerParameters(command=sys.executable, args=["-m", "kb.mcp.server"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            prompt = await s.get_prompt("research", {"topic": "квантование LLM", "depth": "quick"})
            started = _json(await s.call_tool("notebook_write", {"title": "Квантование LLM", "question": "Что даёт 4-bit?"}))
            added = _json(await s.call_tool("notebook_add_source", {
                "section": started["section"], "source_id": "arxiv:2210.17323", "notes": "- **Суть:** GPTQ, 3-4 бита"}))
            invented = _json(await s.call_tool("notebook_add_source", {
                "section": started["section"], "source_id": "arxiv:0000.00000", "notes": "выдумка"}))
            fix = await s.get_prompt("fix_error", {"error": "ValueError: boom", "library": "vllm"})
            overview = _json(await s.call_tool("notebook_read", {}))
            info = _json(await s.call_tool("source_info", {"source_id": "arxiv:2210.17323"}))
            nothing = _json(await s.call_tool("fetch", {}))
            learned = _json(await s.call_tool("notebook_write", {"section": started["section"], "knowledge": [
                {"kind": "practice", "text": "GPTQ квантует по слоям с поправкой ошибки (OBS)",
                 "sources": ["arxiv:2210.17323", "arxiv:0000.00000"]}]}))
            review = _json(await s.call_tool("review_plan", {
                "plan": "Разметка клипов: метки из закрытого списка, размечаю один",
                "section": started["section"]}))
            tried = _json(await s.call_tool("notebook_write", {"section": started["section"], "experiment": {
                "change": "GPTQ 4-bit", "result": "-1.2 MMLU", "verdict": "worse"}}))
            return {"tried": tried, "learned": learned, "review": review, "tools": tools, "prompt": prompt.messages[0].content.text, "added": added, "invented": invented,
                    "fix": fix.messages[0].content.text, "overview": overview, "info": info, "nothing": nothing}


def test_research_flow_over_stdio(tmp_path):
    data = tmp_path / "data"
    db = Database(data / "kb.sqlite")
    db.upsert_paper(Paper(id="arxiv:2210.17323", arxiv_id="2210.17323", title="GPTQ: Accurate Post-Training Quantization",
                          year=2022, venue="ICLR 2023", url="https://arxiv.org/abs/2210.17323"))
    db.commit()
    db.close()
    notebook = tmp_path / "RESEARCH.md"
    env = {**os.environ, "KB_DATA_DIR": str(data), "RESEARCH_NOTEBOOK": str(notebook)}
    out = anyio.run(_run, env)
    assert {"research_sweep", "fetch", "diagnose_error", "check_versions", "notebook_add_source"} <= set(out["tools"])
    assert len(out["tools"]) == 25
    assert "diagnose_error" in out["fix"] and "ValueError: boom" in out["fix"]
    assert "квантование LLM" in out["prompt"] and "notebook_write(title=" in out["prompt"]
    assert out["added"]["status"] == "added"
    assert "error" in out["invented"]                                      # no invented citations
    assert out["overview"]["sections"][0]["papers"] == 1 and out["overview"]["local_base"]["papers"] == 1
    assert out["info"]["in_notebook"] == ["квантование-llm"] and "error" in out["nothing"]
    assert out["tried"]["status"] == "logged" and out["tried"]["failed_or_worse"] == ["GPTQ 4-bit"]
    assert out["learned"]["added"] == 1 and out["learned"]["unknown_sources"] == ["arxiv:0000.00000"]
    assert out["review"]["learned"]["practice"][0]["sources"] == ["arxiv:2210.17323"]
    assert "agreement" in {g["id"] for g in out["review"]["not_addressed"]}
    assert "[GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)" in notebook.read_text()
