from kb.research import reddit
from kb.store import Database

SEARCH = '''
<div class="post" id="abc123">
	<p class="post_header">
		<a class="post_subreddit" href="/r/StableDiffusion">r/StableDiffusion</a>
		<a class="post_author " href="/u/alice">u/alice</a>
		<span class="created" title="Sep 10 2026, 06:14:00 UTC">Sep 10 &#39;26</span>
	</p>
	<h2 class="post_title">
		<a href="/r/StableDiffusion/search?q=flair" class="post_flair" style="x"><span>Question - Help</span></a>
		<a href="/r/StableDiffusion/comments/abc123/gold_standard_speed/">What&#39;s the gold standard for H3 speed?</a>
	</h2>
	<div class="post_score" title="51">51<span class="label"> Upvotes</span></div>
	<div class="post_body post_preview">
		<div class="md"><p>Is comfy kitchen enough?</p>
</div>
	</div>
	<div class="post_footer"><a href="/r/x" class="post_comments" title="55 comments">55 comments</a></div>
</div>
'''

THREAD = '''
<div class="post highlighted">
	<p class="post_header">
		<a class="post_subreddit" href="/r/StableDiffusion">r/StableDiffusion</a>
		<a class="post_author " href="/user/alice">u/alice</a>
		<span class="created" title="Sep 10 2026, 06:14:00 UTC">Sep 10</span>
	</p>
	<h1 class="post_title">
		<a href="/r/StableDiffusion/search?q=f"
			class="post_flair" style="x"><span>Question - Help</span></a>
		What&#39;s the gold standard for H3 speed?
	</h1>
	<div class="post_body"><div class="md"><p>Is comfy kitchen enough?</p></div></div>
	<div class="post_score" title="53">53</div>
	<p id="reddit_url">https://www.reddit.com/r/StableDiffusion/comments/abc123/gold_standard_speed/?sort=top</p>
</div>
<div class="thread">
<div id="c1" class="comment">
	<div class="comment_left"><p class="comment_score" title="14">14</p></div>
	<details class="comment_right" open>
		<summary class="comment_data"><a class="comment_author  " href="/user/bob">u/bob</a>
		<a href="/x" class="created" title="Sep 10 2026, 07:05:12 UTC">Sep 10</a></summary>
		<div class="comment_body "><div class="md"><p>Use <a href="https://github.com/x/ck">ComfyKitchen</a>, not sage.</p>
</div></div>
		<blockquote class="replies">
<div id="c2" class="comment">
	<div class="comment_left"><p class="comment_score" title="2">2</p></div>
	<details class="comment_right" open>
		<summary class="comment_data"><a class="comment_author  " href="/user/alice">u/alice</a>
		<a href="/y" class="created" title="Sep 10 2026, 08:00:00 UTC">Sep 10</a></summary>
		<div class="comment_body "><div class="md"><p>Thanks, 2x faster.</p>
</div></div>
	</details>
</div>
		</blockquote>
	</details>
</div>
</div>
'''


def test_without_redlib_reddit_says_how_to_enable(monkeypatch):
    monkeypatch.delenv("REDLIB_URL", raising=False)
    db = Database()
    try:
        r = reddit.reddit_read(db, "https://www.reddit.com/r/x/comments/abc/t/")
        assert "kb redlib install" in r["next"]
        assert reddit.reddit_search(db, "h3")["results"] == []
    finally:
        db.close()


def test_search_and_thread_parse(monkeypatch):
    monkeypatch.setenv("REDLIB_URL", "http://127.0.0.1:8889")
    pages = {"/search": SEARCH, "/r/StableDiffusion/comments/abc123/gold_standard_speed/": THREAD}
    monkeypatch.setattr(reddit, "_get", lambda path, params=None: pages[path])
    db = Database()
    try:
        s = reddit.reddit_search(db, "h3 speed")["results"][0]
        assert s["id"] == "reddit:abc123" and s["score"] == 51 and s["comments_n"] == 55
        assert s["title"] == "r/StableDiffusion: What's the gold standard for H3 speed?" and s["date"] == "2026-09-10"
        t = reddit.reddit_read(db, "https://old.reddit.com/r/StableDiffusion/comments/abc123/gold_standard_speed/")
        assert t["url"] == "https://www.reddit.com/r/StableDiffusion/comments/abc123/gold_standard_speed/"
        assert t["title"] == "r/StableDiffusion: What's the gold standard for H3 speed?"
        assert "[14] u/bob: Use ComfyKitchen (https://github.com/x/ck), not sage." in t["text"]
        assert "  [2] u/alice (OP): Thanks, 2x faster." in t["text"]
        assert t["comments_n"] == 2 and t["indexed"]["chunks"] >= 1
        assert any("anecdotes" in f for f in t["flags"])
    finally:
        db.close()
