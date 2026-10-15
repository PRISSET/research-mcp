"""arXiv API collector (Atom feed). Docs: https://info.arxiv.org/help/api/user-manual.html"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date

from kb.collect.http import RateLimitedClient
from kb.config import settings
from kb.schemas import Paper

NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
}
_ID_RE = re.compile(r"arxiv\.org/abs/(.+?)(v\d+)?$")
PAGE_SIZE = 200


class ArxivClient:
    def __init__(self) -> None:
        # arXiv throttles bursts with 429 for several minutes; back off patiently.
        self.http = RateLimitedClient(settings.arxiv_min_interval, max_retries=8, max_backoff=300.0)

    def search(
        self, query: str, max_results: int = 200, sort_by: str = "relevance"
    ) -> Iterator[Paper]:
        """Yield papers for an arXiv search query (e.g. 'abs:"video diffusion" AND cat:cs.CV')."""
        start = 0
        while start < max_results:
            batch = min(PAGE_SIZE, max_results - start)
            papers, total = self._fetch(
                {"search_query": query, "start": start, "max_results": batch,
                 "sortBy": sort_by, "sortOrder": "descending"}
            )
            yield from papers
            start += batch
            if not papers or start >= total:
                break

    def by_ids(self, arxiv_ids: list[str]) -> list[Paper]:
        out: list[Paper] = []
        for i in range(0, len(arxiv_ids), 100):
            chunk = arxiv_ids[i : i + 100]
            papers, _ = self._fetch({"id_list": ",".join(chunk), "max_results": len(chunk)})
            out.extend(papers)
        return out

    def _fetch(self, params: dict) -> tuple[list[Paper], int]:
        resp = self.http.get(settings.arxiv_api, params=params)
        root = ET.fromstring(resp.content)
        total_el = root.find("os:totalResults", NS)
        total = int(total_el.text) if total_el is not None and total_el.text else 0
        return [p for e in root.findall("a:entry", NS) if (p := _parse_entry(e))], total

    def close(self) -> None:
        self.http.close()


def _text(el: ET.Element, path: str) -> str:
    found = el.find(path, NS)
    return " ".join((found.text or "").split()) if found is not None else ""


def _parse_entry(e: ET.Element) -> Paper | None:
    m = _ID_RE.search(_text(e, "a:id"))
    title = _text(e, "a:title")
    if not m or not title or title == "Error":
        return None
    arxiv_id = m.group(1)
    published = date.fromisoformat(_text(e, "a:published")[:10])
    pdf = next((l.get("href") for l in e.findall("a:link", NS) if l.get("title") == "pdf"), None)
    return Paper(
        id=f"arxiv:{arxiv_id}",
        arxiv_id=arxiv_id,
        title=title,
        abstract=_text(e, "a:summary"),
        authors=[_text(a, "a:name") for a in e.findall("a:author", NS)],
        year=published.year,
        published=published,
        updated=date.fromisoformat(_text(e, "a:updated")[:10]),
        doi=_text(e, "arxiv:doi") or None,
        categories=[c.get("term", "") for c in e.findall("a:category", NS)],
        venue=_text(e, "arxiv:journal_ref") or None,
        comment=_text(e, "arxiv:comment") or None,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=pdf,
        source="arxiv",
    )
