"""Title normalisation and fuzzy matching shared by the collectors."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def normalize(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()
