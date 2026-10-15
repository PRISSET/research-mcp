"""Project paths and settings. Override any value with a KB_* environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"KB_{name}", default)


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    taxonomy_dir: Path = ROOT / "taxonomy"
    seeds_dir: Path = ROOT / "seeds"

    # Polite crawling: arXiv asks for >= 3 s between API calls.
    arxiv_api: str = "https://export.arxiv.org/api/query"
    arxiv_min_interval: float = float(_env("ARXIV_INTERVAL", "3.1"))
    s2_api: str = "https://api.semanticscholar.org/graph/v1"
    s2_api_key: str | None = os.environ.get("S2_API_KEY")
    s2_min_interval: float = float(_env("S2_INTERVAL", "1.1"))
    user_agent: str = "research-mcp/0.2 (+https://github.com/PRISSET/research-mcp)"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "kb.sqlite"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"


settings = Settings()
