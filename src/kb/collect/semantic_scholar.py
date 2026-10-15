"""Semantic Scholar Graph API: citation counts, TL;DRs, venues, title matching.

Set S2_API_KEY for higher rate limits (free: https://www.semanticscholar.org/product/api).
"""

from __future__ import annotations

from typing import Any

import httpx

from kb.collect.http import RateLimitedClient
from kb.config import settings

PAPER_FIELDS = "paperId,externalIds,title,venue,year,citationCount,influentialCitationCount,tldr,publicationTypes"
BATCH_SIZE = 400


class SemanticScholarClient:
    def __init__(self) -> None:
        headers = {"x-api-key": settings.s2_api_key} if settings.s2_api_key else {}
        self.http = RateLimitedClient(settings.s2_min_interval, headers=headers, max_retries=8)

    def batch(self, ids: list[str], fields: str = PAPER_FIELDS) -> list[dict[str, Any] | None]:
        """Look up papers by ids like 'ARXIV:2401.12345' or 'DOI:...'. Result is aligned with input."""
        out: list[dict[str, Any] | None] = []
        for i in range(0, len(ids), BATCH_SIZE):
            chunk = ids[i : i + BATCH_SIZE]
            resp = self.http.post(f"{settings.s2_api}/paper/batch", params={"fields": fields}, json={"ids": chunk})
            out.extend(resp.json())
        return out

    def match_title(self, title: str) -> dict[str, Any] | None:
        try:
            resp = self.http.get(
                f"{settings.s2_api}/paper/search/match", params={"query": title, "fields": PAPER_FIELDS}
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        data = resp.json().get("data") or []
        return data[0] if data else None

    def close(self) -> None:
        self.http.close()


def to_db_fields(item: dict[str, Any]) -> dict[str, Any]:
    import json

    ext = item.get("externalIds") or {}
    return {
        "s2_id": item.get("paperId"),
        "citation_count": item.get("citationCount"),
        "influential_count": item.get("influentialCitationCount"),
        "tldr": (item.get("tldr") or {}).get("text"),
        "venue": item.get("venue") or None,
        "doi": ext.get("DOI"),
        "publication_types": json.dumps(item.get("publicationTypes") or []),
    }
