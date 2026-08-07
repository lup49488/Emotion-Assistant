"""Source filtering shared by the knowledge and style RAG stores.

Both stores search one flat index covering every document, so a controlled
experiment needs a way to restrict retrieval to a chosen subset (one knowledge
topic, one style family). The matching rule lives here rather than in each store
so the two can never drift apart and silently produce different conditions.

Filters match the file name stem, case-insensitively: ``"温柔型"`` selects
``温柔型.md`` together with every ``温柔型_*.md`` sub-style, while a full stem
selects exactly that document. An empty filter means no restriction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence


def normalize_prefix(source_prefix: str | None) -> str:
    return (source_prefix or "").strip().casefold()


def matches_prefix(source: Any, prefix: str) -> bool:
    if not prefix:
        return True
    return Path(str(source or "")).stem.casefold().startswith(prefix)


def family_prefixes(documents: Iterable[str]) -> list[str]:
    """Selectable families derived from the file name.

    Documents are named ``<family>.md`` or ``<family>_<variant>.md``, so the text
    before the first underscore identifies the family.
    """
    prefixes = {Path(name).stem.split("_", 1)[0].strip() for name in documents}
    return sorted(prefix for prefix in prefixes if prefix)


def resolve_sources(documents: Iterable[str], source_prefix: str | None) -> list[str]:
    """Document names a filter actually selects, for verifying an experiment run."""
    prefix = normalize_prefix(source_prefix)
    return sorted(name for name in documents if matches_prefix(name, prefix))


def allowed_positions(
    chunks: Sequence[dict[str, Any]], prefix: str
) -> set[int] | None:
    """Chunk positions surviving the filter, or ``None`` when nothing is filtered.

    ``None`` rather than "every position" so callers can skip widening their
    candidate pool on the common unfiltered path.
    """
    if not prefix:
        return None
    return {
        position
        for position, chunk in enumerate(chunks)
        if matches_prefix(chunk.get("source", ""), prefix)
    }
