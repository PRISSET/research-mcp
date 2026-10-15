"""Storage layer: SQLite metadata + FTS5 (vector index is added in the processing stage)."""

from kb.store.db import Database

__all__ = ["Database"]
