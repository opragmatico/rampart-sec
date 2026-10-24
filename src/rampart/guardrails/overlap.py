"""Verbatim-overlap check: share of the answer's word n-grams that also occur in retrieved chunks."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")


def _ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    w = _WORD.findall(text.lower())
    return [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]


def overlap_ratio(answer: str, sources: list[str], n: int = 10) -> float:
    grams = _ngrams(answer, n)
    if not grams:
        return 0.0
    pool: set[tuple[str, ...]] = set()
    for s in sources:
        pool.update(_ngrams(s, n))
    return sum(g in pool for g in grams) / len(grams)
