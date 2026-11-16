import json
import re
import zlib
from contextlib import nullcontext

import numpy as np
import pytest

from kb import notebook
from kb.config import Settings
from kb.memory import digest, hook, llm, store
from kb.store import Database


def _fake_vectors(texts):
    """Bag of 4-letter word stems hashed into 64 dims: close texts get close vectors, no model download."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            out[i, zlib.crc32(w[:4].encode()) % 64] += 1
    return (out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)).astype(np.float16)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr("kb.store.db.settings", Settings(data_dir=tmp_path / "data"))
    monkeypatch.setattr(store, "_vectors", _fake_vectors)
    monkeypatch.setenv("RESEARCH_NOTEBOOK", str(tmp_path / "RESEARCH.md"))
    monkeypatch.setenv("KB_MEMORY_DIR", str(tmp_path / "memory"))
    with Database() as d:
        yield d


def test_same_fact_updates_instead_of_duplicating(db):
    assert store.add(db, "ComfyUI", "Видеокарта RTX 3060 Ti 8 ГБ занята ComfyUI")["action"] == "added"
    again = store.add(db, "comfyui", "видеокарта rtx 3060 ti 8 гб занята comfyui")
    assert again["action"] == "updated"
    assert store.add(db, "ComfyUI", "Wan 2.2 рендерит 81 кадр за 6 минут")["action"] == "added"
    assert db.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
    with pytest.raises(ValueError):
        store.add(db, "", "x")


def test_recall_finds_other_word_forms_and_keeps_the_budget(db):
    store.add(db, "LoRA", "Для Flux LoRA лучший learning rate 1e-4 при rank 16")
    store.add(db, "Видео", "Wan 2.2 рендерит 81 кадр за 6 минут на 3060 Ti")
    for i in range(30):
        store.add(db, f"шум {i}", f"посторонняя заметка номер {i} " + "текст " * 40)
    out = store.recall(db, "какой learning rate для лоры flux", budget_tokens=200)
    assert out["notes"][0]["topic"] == "LoRA" and out["tokens"] <= 200
    assert store.recall(db, "рендерит кадры")["notes"][0]["topic"] == "Видео"   # "кадры" meets "кадр"
    only = store.recall(db, "заметка", topic="шум 3")["notes"]
    assert [n["topic"] for n in only] == ["шум 3"]
    assert store.forget(db, [only[0]["id"]]) == 1


def test_nothing_found_lists_the_topics(db):
    store.add(db, "LoRA", "rank 16")
    out = store.recall(db, "квантовая хромодинамика")
    assert out["notes"] == [] and out["topics"][0]["topic"] == "LoRA"


def test_notebook_sections_are_mirrored_and_follow_changes(db):
    sec = notebook.start("Апскейл видео", "чем апскейлить Wan")["section"]
    notebook.write(sec, "SeedVR2 даёт лучший апскейл до 1080p, но требует 12 ГБ VRAM.")
    assert store.sync_notebook(db) == {"sections": 1, "changed": 1, "removed": 0}
    assert store.sync_notebook(db)["changed"] == 0          # unchanged: nothing re-embedded
    hit = store.recall(db, "апскейл SeedVR2")["notes"][0]
    assert hit["origin"] == "notebook" and "<!--" not in hit["text"]
    notebook.write(sec, "SeedVR2 на 8 ГБ работает с block swap.")
    assert store.sync_notebook(db)["changed"] == 1
    assert "block swap" in store.recall(db, "SeedVR2 block swap")["notes"][0]["text"]


def _transcript(path, entries):
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n")
    return path


def _msg(kind, content, **extra):
    return {"type": kind, "message": {"role": kind, "content": content}, **extra}


def test_only_the_conversation_is_read(tmp_path):
    path = _transcript(tmp_path / "s.jsonl", [
        _msg("user", "<system-reminder>rules</system-reminder>Сделай LoRA для Flux"),
        _msg("assistant", [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Беру rank 16"},
                           {"type": "tool_use", "name": "Bash", "input": {}}]),
        _msg("user", [{"type": "tool_result", "content": "huge output"}]),
        _msg("user", "meta", isMeta=True),
        {"type": "summary", "summary": "x"},
        "not json",
    ])
    messages, n = digest.read_chat(path)
    assert messages == [(1, "User: Сделай LoRA для Flux"), (2, "Assistant: Беру rank 16")] and n == 6
    assert digest.read_chat(path, 1)[0] == [(2, "Assistant: Беру rank 16")]


def test_parse_notes_skips_broken_items():
    assert digest.parse_notes('ok {"notes": [{"topic": "A", "text": "b"}, {"topic": " "}, 3]}') == [
        {"topic": "A", "text": "b"}]
    assert digest.parse_notes('{"notes": []}') == []
    with pytest.raises(digest.BadAnswer):
        digest.parse_notes("I cannot help with that")


def test_digest_writes_notes_and_continues_where_it_stopped(db, tmp_path, monkeypatch):
    prompts = []

    def chat(messages, schema=None, **kw):
        prompts.append(messages[0]["content"])
        assert schema["properties"]["notes"]["type"] == "array"
        return json.dumps({"notes": [{"topic": "NSFW видео", "text": "Для откровенных сцен берём Wan 2.2 без "
                                                                     "цензуры, CFG 3.5"}]}, ensure_ascii=False)

    monkeypatch.setattr(llm, "running", nullcontext)
    monkeypatch.setattr(llm, "chat", chat)
    store.add(db, "LoRA", "rank 16")
    path = _transcript(tmp_path / "s.jsonl", [_msg("user", "Что ставить для откровенных сцен? " * 20),
                                              _msg("assistant", "Wan 2.2, CFG 3.5 " * 30)])
    out = digest.digest(path, "s1", "/proj")
    assert out == {"session": "s1", "notes": 1, "added": 1, "updated": 0}
    assert "LoRA" in prompts[0] and "do not refuse" in prompts[0]
    row = db.conn.execute("SELECT origin, ref, project FROM memories WHERE topic = 'NSFW видео'").fetchone()
    assert tuple(row) == ("chat", "s1", "/proj")
    assert digest.digest(path, "s1")["skipped"]                     # no new lines since
    assert len(prompts) == 1


def test_long_chats_are_digested_part_by_part(monkeypatch):
    monkeypatch.setattr(digest, "MAX_CHARS", 100)
    parts = digest.portions([(i + 1, f"User: message {i} " + "x" * 30) for i in range(10)])
    assert len(parts) == 5 and parts[-1][0] == 10
    assert "message 0" in parts[0][1] and "message 9" in parts[-1][1]


def test_a_bad_answer_keeps_the_position(db, tmp_path, monkeypatch):
    replies = ["not json", '{"notes": []}']
    monkeypatch.setattr(llm, "running", nullcontext)
    monkeypatch.setattr(llm, "chat", lambda messages, schema=None, **kw: replies.pop(0))
    path = _transcript(tmp_path / "s.jsonl", [_msg("user", "Что ставить? " * 60)])
    with pytest.raises(digest.BadAnswer):
        digest.digest(path, "s2")
    assert digest.digest(path, "s2")["notes"] == 0 and not replies      # retried, then done


def test_notebook_mirror_is_not_overwritten_by_chat_notes(db):
    sec = notebook.start("LoRA", "lr")["section"]
    notebook.write(sec, "rank 16 lr 1e-4")
    store.sync_notebook(db)
    assert store.add(db, "LoRA", "rank 16 lr 1e-4", origin="chat")["action"] == "added"


def test_one_long_note_is_cut_to_the_budget(db):
    store.add(db, "Длинное", "слово " * 2000)
    note, = store.recall(db, "слово", budget_tokens=300)["notes"]
    assert len(note["text"]) < 300 * store.CHARS_PER_TOKEN


def test_chat_asks_the_server_for_json(monkeypatch):
    sent = {}

    class Reply:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"notes": []}'}}]}

    monkeypatch.setattr(llm.httpx, "post", lambda url, json, timeout: (sent.update(url=url, body=json), Reply)[1])
    assert llm.chat([{"role": "user", "content": "q"}], schema={"type": "object"}) == '{"notes": []}'
    assert sent["url"].endswith("/v1/chat/completions")
    assert sent["body"]["response_format"]["json_schema"]["schema"] == {"type": "object"}


def test_missing_model_says_how_to_install(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("KB_MEMORY_PORT", "9")
    monkeypatch.setenv("KB_LLAMA_SERVER", "/bin/true")
    with pytest.raises(llm.Unavailable, match="kb memory install"), llm.running():
        pass


def test_session_start_lists_topics(db):
    assert hook.hook("session", {}) is None
    store.add(db, "LoRA", "rank 16")
    ctx = hook.hook("session", {})["hookSpecificOutput"]["additionalContext"]
    assert "LoRA (1)" in ctx and "recall(" in ctx


def test_end_starts_a_background_digest(db, monkeypatch):
    started = []
    monkeypatch.setattr(hook.subprocess, "Popen", lambda cmd, **kw: started.append((cmd, kw)))
    hook.hook("end", {"transcript_path": "/t.jsonl", "session_id": "s1", "cwd": "/p"})
    hook.hook("end", {})
    (cmd, kw), = started
    assert cmd[-4:] == ["kb.memory.digest", "/t.jsonl", "s1", "/p"] and kw["start_new_session"]


def test_hooks_install_next_to_the_guard(tmp_path, monkeypatch):
    from kb import guard

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    guard.install()
    hook.install()
    hook.install()
    cfg = json.loads((tmp_path / "settings.json").read_text())
    assert len(cfg["hooks"]["SessionStart"]) == 2 and len(cfg["hooks"]["SessionEnd"]) == 1
    assert hook.uninstall() and not hook.uninstall()
    cfg = json.loads((tmp_path / "settings.json").read_text())
    assert "SessionEnd" not in cfg["hooks"] and len(cfg["hooks"]["SessionStart"]) == 1


def test_workflows_are_found_by_goal_with_their_path(db, tmp_path):
    from kb.memory import workflows as wf

    fast = tmp_path / "wf" / "speed"
    fast.mkdir(parents=True)
    ui = {"last_node_id": 3, "links": [], "nodes": [
        {"type": "DPCacheApply", "widgets_values": [0.2]},
        {"type": "KSampler", "widgets_values": [1, "fixed", 8, 1.0, "euler", "simple", 1.0]},
        {"type": "UNETLoader", "widgets_values": ["minimax_h3_turbo.safetensors", "default"]},
        {"type": "Note", "widgets_values": ["Быстрый режим: 8 шагов, DPcache, качество почти как 30 шагов"]}]}
    (fast / "h3_fast.json").write_text(json.dumps(ui, ensure_ascii=False))
    api = {"1": {"class_type": "SeedVR2Upscale", "inputs": {"width": 1920, "height": 1080, "model": ["2", 0]},
                 "_meta": {"title": "Upscale to 1080p"}},
           "2": {"class_type": "SeedVR2LoadModel", "inputs": {"model": "seedvr2_ema_7b_fp8_e4m3fn.safetensors"}}}
    (tmp_path / "wf" / "upscale.json").write_text(json.dumps(api))
    (tmp_path / "wf" / "package.json").write_text(json.dumps({"name": "x", "version": "1", "nodes": "no" * 200}))
    store.add(db, "H3 скорость", "DPcache на 8 шагах дал 2x ускорение, лица чуть мягче")
    assert wf.scan(db, [tmp_path / "wf"]) == {"workflows": 2, "changed": 3, "removed": 0}   # package.json read once
    assert wf.scan(db, [tmp_path / "wf"])["changed"] == 0
    text = wf.describe(fast / "h3_fast.json")
    assert "minimax_h3_turbo.safetensors" in text and "KSampler[1, fixed, 8, 1, euler" in text
    assert "DPCacheApply" in text and "Быстрый режим" in text
    assert "width=1920" in wf.describe(tmp_path / "wf" / "upscale.json")
    out = wf.search(db, "быстрый режим DPcache", rescan=False)
    assert out["workflows"][0]["path"] == str(fast / "h3_fast.json")
    assert out["recorded"][0]["topic"] == "H3 скорость"
    (tmp_path / "wf" / "upscale.json").unlink()
    assert wf.scan(db, [tmp_path / "wf"])["removed"] == 1


def test_broken_workflow_files_do_not_stop_the_scan(db, tmp_path):
    from kb.memory import workflows as wf

    d = tmp_path / "wf"
    d.mkdir()
    (d / "bad.json").write_text(json.dumps({"nodes": [None, 3, "x"], "links": [], "definitions": []}) + " " * 60)
    (d / "ok.json").write_text(json.dumps({"1": {"class_type": "KSampler", "inputs": {"steps": 8}}}))
    assert wf.scan(db, [d])["workflows"] == 1
    assert wf.scan(db, [tmp_path / "elsewhere"])["removed"] == 0        # files still there keep their notes
    assert db.conn.execute("SELECT COUNT(*) FROM memories WHERE origin = 'workflow'").fetchone()[0] == 1
