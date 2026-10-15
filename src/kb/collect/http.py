"""HTTP client with per-host rate limiting and retry/backoff on 429 / 5xx."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from kb.config import settings

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class RateLimitedClient:
    def __init__(
        self,
        min_interval: float,
        headers: dict[str, str] | None = None,
        max_retries: int = 6,
        max_backoff: float = 60.0,
        use_cache: bool = True,
        timeout: float = 60.0,
    ) -> None:
        self.use_cache = use_cache
        # A 200 is not always an answer: search engines reply "no results" to clients they think are bots.
        self.cacheable: Any = lambda resp: True
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.max_backoff = max_backoff
        self._last = 0.0
        self._client = httpx.Client(
            headers={"User-Agent": settings.user_agent, **(headers or {})},
            timeout=httpx.Timeout(timeout, connect=min(15.0, timeout)),
            follow_redirects=True,
        )

    def _wait(self) -> None:
        delay = self.min_interval - (time.monotonic() - self._last)
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        from kb.collect import cache

        key = None
        if self.use_cache and cache.enabled():
            req = self._client.build_request(method, url, **{k: v for k, v in kwargs.items()
                                                             if k in ("params", "json", "data", "content")})
            key = cache.key_for(method, str(req.url), req.read() if method.upper() != "GET" else None)
            hit = cache.get(key)
            if hit is not None:
                status, headers, body = hit
                return httpx.Response(status, headers=headers, content=body, request=req)
        resp = self._request(method, url, **kwargs)
        if key is not None and resp.status_code == 200 and self.cacheable(resp):
            cache.put(key, str(resp.request.url), 200, dict(resp.headers), resp.content, cache.ttl_for(str(resp.url)))
        return resp

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            self._wait()
            try:
                resp = self._client.request(method, url, **kwargs)
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise
                log.warning("transport error %s, retry %d", e, attempt + 1)
                time.sleep(2**attempt)
                continue
            if resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                resp.raise_for_status()
                return resp
            retry_after = resp.headers.get("Retry-After")
            # Retry-After is honoured but capped: an interactive call must not block the agent for a minute.
            wanted = float(retry_after) if retry_after and retry_after.isdigit() else 5.0 * 2**attempt
            backoff = min(self.max_backoff, wanted)
            if wanted > self.max_backoff * 3 and self.max_retries <= 2:   # long ban: give up now, others cover
                resp.raise_for_status()
            log.warning("HTTP %d from %s, sleeping %.0fs", resp.status_code, url.split("?")[0], backoff)
            time.sleep(backoff)
        raise RuntimeError("unreachable")

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def close(self) -> None:
        self._client.close()
