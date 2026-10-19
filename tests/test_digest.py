from kb.research import digest


def test_topic_query_prefers_saved_query_then_model_names():
    assert digest.topic_query({"title": "Что угодно", "queries": ["video action annotation"]}) == (
        "video action annotation", True)
    assert digest.topic_query({"title": "Ускорение MiniMax H3 без потери качества", "papers": []})[0] == "MiniMax H3"
    assert digest.topic_query({"title": "LoRA персонажа на Qwen-Image 2.1 (8 ГБ)"})[0] == "Qwen-Image 2.1"
    # a bare "H3" is widened by the family the topic cites
    assert digest.topic_query({"title": "Анатомия на H3", "papers": ["hf:someone/minimax-h3-loras"]})[0] == "minimax h3"
    assert digest.topic_query({"title": "Звуковая разметка", "papers": ["gh:x/voxcpm", "hf:qwen/qwen3-omni"]})[0] == "qwen3"


def test_build_dedupes_shared_queries_and_take_shows_news_once(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "_path", lambda: tmp_path / "digest.json")
    sections = [{"id": "a", "title": "Ускорение MiniMax H3", "updated": "2026-10-18"},
                {"id": "b", "title": "Анатомия MiniMax H3", "updated": "2026-10-17"}]
    monkeypatch.setattr(digest.notebook, "overview", lambda: {"sections": sections})
    monkeypatch.setattr(digest.notebook, "read_section", lambda sid: next(s for s in sections if s["id"] == sid))
    calls = []

    def fake(sec, since, query, research):
        calls.append((sec["id"], query))
        return [{"kind": "model", "title": "Org/New", "url": "https://huggingface.co/Org/New", "date": "2026-10-18",
                 "why": "new model"}] if query else []

    monkeypatch.setattr(digest, "_section_news", fake)
    digest.build()
    assert calls == [("a", "MiniMax H3"), ("b", "")]   # one search for both topics
    assert digest.pending() and not digest.due()
    news = digest.take()
    assert list(news["by_topic"]) == ["a"] and news["by_topic"]["a"][0]["title"] == "Org/New"
    assert digest.take() is None
    digest.build()   # the same item found again is not news
    assert digest.take() is None
