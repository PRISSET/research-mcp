"""Title normalisation and fuzzy matching shared by the collectors."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def normalize(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


LICENSE_HEAD = re.compile(r"(?im)^(#{1,6})\s*[^\n]*\b(licen[cs]e|licensing)\b[^\n]*$")
LICENSE_YAML = re.compile(r"(?im)^(license|license_name|license_link)\s*:.*\n?")
LICENSE_BADGE = re.compile(r"(?i)<a [^>]*>\s*<img [^>]*licen[cs]e[^>]*>\s*</a>|\[!\[[^\]]*licen[cs]e[^\]]*\]\([^)]*\)\]\([^)]*\)")


def drop_licenses(text: str) -> str:
    """Model cards and READMEs without their licence parts: YAML license fields, badges and "License" sections
    (up to the next heading of the same or a higher level). The user does not want tokens spent on them."""
    text = LICENSE_BADGE.sub("", LICENSE_YAML.sub("", text))
    while m := LICENSE_HEAD.search(text):
        level = len(m.group(1))
        nxt = re.compile(rf"(?m)^#{{1,{level}}}\s").search(text, m.end())
        text = text[:m.start()] + (text[nxt.start():] if nxt else "")
    return text
