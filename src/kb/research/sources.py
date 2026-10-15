"""Live paper search in OpenAlex (all fields), arXiv and Semantic Scholar. No keys needed.

OpenAlex without a key: about 100 searches a day (a search costs 10 of 1000 daily credits, single-work lookups
are free); OPENALEX_API_KEY raises that tenfold. Semantic Scholar without S2_API_KEY often answers 429: it is
best effort. Interactive calls retry briefly and give up instead of blocking the agent for minutes.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date
from typing import Any

from kb.collect.arxiv import ArxivClient
from kb.collect.http import RateLimitedClient
from kb.config import settings
from kb.schemas import Paper

log = logging.getLogger(__name__)

OPENALEX = "https://api.openalex.org"
OA_SELECT = ("id,doi,display_name,publication_year,publication_date,authorships,abstract_inverted_index,"
             "cited_by_count,primary_location,best_oa_location,type")
S2_SEARCH_FIELDS = ("paperId,externalIds,title,abstract,year,venue,citationCount,influentialCitationCount,tldr,"
                    "openAccessPdf,publicationDate,authors")
ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)
ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(?:v\d+)?(?:\.pdf)?$", re.IGNORECASE)
STOP = {"the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "with", "by", "from", "via", "is", "are", "how",
        "what", "using", "based", "towards", "toward", "into", "at", "as", "its", "their", "vs"}


INTERACTIVE_TIMEOUT = 20.0   # seconds per request: an agent is waiting


def interactive_client(min_interval: float, headers: dict[str, str] | None = None) -> RateLimitedClient:
    return RateLimitedClient(min_interval, headers=headers, max_retries=2, max_backoff=8.0, timeout=INTERACTIVE_TIMEOUT)


# ---------------- OpenAlex ----------------

def _abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    words = sorted((pos, w) for w, positions in inv.items() for pos in positions)
    return " ".join(w for _, w in words)


def _arxiv_from(*candidates: str | None) -> str | None:
    for c in candidates:
        if not c:
            continue
        m = ARXIV_DOI.match(c) or ARXIV_URL.search(c)
        if m:
            return m.group(1)
    return None


def parse_openalex(w: dict[str, Any]) -> Paper | None:
    title = " ".join((w.get("display_name") or "").split())
    if not title:
        return None
    oa_id = (w.get("id") or "").rsplit("/", 1)[-1] or None
    doi = (w.get("doi") or "").removeprefix("https://doi.org/") or None
    best = w.get("best_oa_location") or {}
    primary = w.get("primary_location") or {}
    arxiv_id = _arxiv_from(doi, best.get("landing_page_url"), best.get("pdf_url"), primary.get("landing_page_url"))
    source = (primary.get("source") or {})
    venue = source.get("display_name") if source.get("type") != "repository" else None
    published = None
    if w.get("publication_date"):
        try:
            published = date.fromisoformat(w["publication_date"][:10])
        except ValueError:
            pass
    if arxiv_id:
        pid, url = f"arxiv:{arxiv_id}", f"https://arxiv.org/abs/{arxiv_id}"
    elif doi:
        pid, url = f"doi:{doi.lower()}", f"https://doi.org/{doi}"
    else:
        pid, url = f"openalex:{oa_id}", primary.get("landing_page_url") or f"https://openalex.org/{oa_id}"
    return Paper(
        id=pid, title=title, abstract=_abstract(w.get("abstract_inverted_index")),
        authors=[(a.get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])][:30],
        year=w.get("publication_year"), published=published, arxiv_id=arxiv_id,
        doi=None if doi and ARXIV_DOI.match(doi) else doi, openalex_id=oa_id, venue=venue, url=url,
        pdf_url=best.get("pdf_url") or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None),
        citation_count=w.get("cited_by_count"), categories=[w["type"]] if w.get("type") else [], source="openalex",
    )


class OpenAlex:
    def __init__(self) -> None:
        self.key = os.environ.get("OPENALEX_API_KEY")
        self.http = interactive_client(0.15)

    def _params(self, **kw: Any) -> dict[str, Any]:
        return {**kw, **({"api_key": self.key} if self.key else {})}

    def search(self, query: str, limit: int = 25, year_from: int | None = None) -> list[Paper]:
        params = self._params(search=query, select=OA_SELECT, **{"per-page": min(limit, 100)})
        if year_from:
            params["filter"] = f"from_publication_date:{year_from}-01-01"
        data = self.http.get(f"{OPENALEX}/works", params=params).json()
        return [p for w in data.get("results", []) if (p := parse_openalex(w))]

    def work(self, ref: str) -> dict[str, Any] | None:
        """Raw work by OpenAlex id or "doi:10.…" (free lookup)."""
        try:
            return self.http.get(f"{OPENALEX}/works/{ref}", params=self._params()).json()
        except Exception as e:
            log.info("openalex work %s: %s", ref, e)
            return None

    def works_filter(self, flt: str, limit: int = 25, sort: str | None = "cited_by_count:desc") -> list[Paper]:
        params = self._params(filter=flt, select=OA_SELECT, **{"per-page": min(limit, 100)})
        if sort:
            params["sort"] = sort
        data = self.http.get(f"{OPENALEX}/works", params=params).json()
        return [p for w in data.get("results", []) if (p := parse_openalex(w))]

    def close(self) -> None:
        self.http.close()


# ---------------- Semantic Scholar ----------------

def parse_s2(item: dict[str, Any]) -> Paper | None:
    title = " ".join((item.get("title") or "").split())
    if not title or not item.get("paperId"):
        return None
    ext = item.get("externalIds") or {}
    arxiv_id, doi = ext.get("ArXiv"), ext.get("DOI")
    published = None
    if item.get("publicationDate"):
        try:
            published = date.fromisoformat(item["publicationDate"])
        except ValueError:
            pass
    if arxiv_id:
        pid, url = f"arxiv:{arxiv_id}", f"https://arxiv.org/abs/{arxiv_id}"
    elif doi:
        pid, url = f"doi:{doi.lower()}", f"https://doi.org/{doi}"
    else:
        pid, url = f"s2:{item['paperId']}", f"https://www.semanticscholar.org/paper/{item['paperId']}"
    return Paper(
        id=pid, title=title, abstract=item.get("abstract") or "",
        authors=[a.get("name", "") for a in (item.get("authors") or [])][:30], year=item.get("year"),
        published=published, arxiv_id=arxiv_id, doi=None if doi and ARXIV_DOI.match(doi) else doi,
        s2_id=item["paperId"], venue=item.get("venue") or None, url=url,
        pdf_url=(item.get("openAccessPdf") or {}).get("url") or None, citation_count=item.get("citationCount"),
        influential_count=item.get("influentialCitationCount"), tldr=(item.get("tldr") or {}).get("text"),
        source="s2",
    )


class SemanticScholar:
    def __init__(self) -> None:
        headers = {"x-api-key": settings.s2_api_key} if settings.s2_api_key else {}
        self.http = interactive_client(settings.s2_min_interval, headers)
        if not settings.s2_api_key:  # the shared keyless pool is often exhausted: fail fast, other sources cover
            self.http.max_retries, self.http.max_backoff = 1, 3.0

    def search(self, query: str, limit: int = 25, year_from: int | None = None) -> list[Paper]:
        params: dict[str, Any] = {"query": query, "limit": min(limit, 100), "fields": S2_SEARCH_FIELDS}
        if year_from:
            params["year"] = f"{year_from}-"
        data = self.http.get(f"{settings.s2_api}/paper/search", params=params).json()
        return [p for item in data.get("data") or [] if (p := parse_s2(item))]

    def links(self, ref: str, direction: str, limit: int = 50) -> list[tuple[Paper, bool]]:
        """References or citations of a paper: [(paper, is_influential)]."""
        key = "citedPaper" if direction == "references" else "citingPaper"
        fields = "isInfluential," + ",".join(f"{f}" for f in S2_SEARCH_FIELDS.split(",") if f != "tldr")
        data = self.http.get(f"{settings.s2_api}/paper/{ref}/{direction}",
                             params={"fields": fields, "limit": min(limit, 1000)}).json()
        out = []
        for row in data.get("data") or []:
            p = parse_s2(row.get(key) or {})
            if p:
                out.append((p, bool(row.get("isInfluential"))))
        return out

    def close(self) -> None:
        self.http.close()


# ---------------- arXiv ----------------

def arxiv_query(query: str, max_terms: int = 6) -> str:
    terms = [t for t in re.findall(r"[\w][\w\-]*", query.lower()) if t not in STOP and len(t) > 1][:max_terms]
    return " AND ".join(f"all:{t}" for t in terms)


def _interactive_arxiv() -> ArxivClient:
    import httpx

    client = ArxivClient()
    client.http.max_retries, client.http.max_backoff = 1, 6.0
    client.http._client.timeout = httpx.Timeout(INTERACTIVE_TIMEOUT, connect=10.0)
    return client


def arxiv_search(query: str, limit: int = 25, year_from: int | None = None) -> list[Paper]:
    client = _interactive_arxiv()
    try:
        out: list[Paper] = []
        for n_terms in (6, 3):  # a strict AND over many terms can return nothing: retry with the head terms
            q = arxiv_query(query, n_terms)
            if not q:
                return []
            out = [p for p in client.search(q, max_results=limit) if not year_from or (p.year or 0) >= year_from]
            if out:
                break
        return out
    finally:
        client.close()


def arxiv_by_id(arxiv_id: str) -> Paper | None:
    client = _interactive_arxiv()
    try:
        found = client.by_ids([arxiv_id])
        return found[0] if found else None
    finally:
        client.close()
