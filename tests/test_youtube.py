"""YouTube search / read parsing — no network."""

import json

from kb.research import youtube as yt


class _Http:
    def __init__(self, html):
        self.html = html

    def get(self, url, params=None):
        return type("R", (), {"text": self.html})()

    def close(self):
        pass


def test_search_parses_video_renderers(monkeypatch, tmp_path):
    data = {"contents": [{"videoRenderer": {
        "videoId": "DRAFkmsCwKw", "title": {"runs": [{"text": "3x Faster MiniMax H3"}]},
        "ownerText": {"runs": [{"text": "Veteran AI"}]}, "viewCountText": {"simpleText": "12,295 views"},
        "lengthText": {"simpleText": "10:22"}, "publishedTimeText": {"simpleText": "2 months ago"}}}]}
    html = f"<script>var ytInitialData = {json.dumps(data)};</script>"
    monkeypatch.setattr(yt, "interactive_client", lambda *a, **k: _Http(html))
    from kb.store import Database

    r = yt.youtube_search(Database(tmp_path / "kb.sqlite"), "minimax h3", 3)["results"][0]
    assert r["id"] == "yt:DRAFkmsCwKw" and r["views"] == 12295 and r["channel"] == "Veteran AI"
    assert any("check settings" in f for f in r["flags"])


def test_read_builds_timed_paragraphs_and_chapters(monkeypatch, tmp_path):
    player = {"videoDetails": {"title": "Speed stack", "author": "Veteran AI", "viewCount": "100", "lengthSeconds": "130",
                               "shortDescription": "Nodes:\n0:00 Intro\n1:00 EasyCache settings"},
              "microformat": {"playerMicroformatRenderer": {"publishDate": "2026-08-07T04:26:26-07:00"}}}
    html = f"<script>var ytInitialPlayerResponse = {json.dumps(player)};var x=1</script>"
    monkeypatch.setattr(yt, "interactive_client", lambda *a, **k: _Http(html))
    monkeypatch.setattr(yt, "_transcript", lambda vid: ([(0.5, "hello"), (20, "we compare"), (61, "reuse_threshold 0.3"),
                                                        (70, "is the sweet spot")], "en"))
    from kb.store import Database

    r = yt.youtube_read(Database(tmp_path / "kb.sqlite"), "https://youtu.be/DRAFkmsCwKw?t=5")
    assert r["published"] == "2026-08-07" and r["chapters"] == ["0:00 Intro", "1:00 EasyCache settings"]
    assert "## 1:00 EasyCache settings\n\n[1:01] reuse_threshold 0.3 is the sweet spot" in r["text"]
    assert "[0:00] hello we compare" in r["text"] and r["indexed"]["chunks"] >= 1
