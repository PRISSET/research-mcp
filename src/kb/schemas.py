"""Core data records shared across collectors, storage and the MCP layer."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class Paper(BaseModel):
    id: str                                   # canonical id, e.g. "arxiv:2401.12345"
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    published: date | None = None
    updated: date | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None
    categories: list[str] = Field(default_factory=list)
    venue: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    comment: str | None = None                # arXiv author comment, often holds venue / project page
    citation_count: int | None = None
    influential_count: int | None = None
    tldr: str | None = None
    source: str = "arxiv"                     # arxiv | openalex | s2
