from __future__ import annotations

from rag.baseline import MIN_SCORE
from rag.common import Chunk, Hit
from rag.fixed import filter_hits


def test_filter_keeps_low_scoring_hits_instead_of_dropping_them() -> None:
    hits = [Hit(Chunk("d", "ar", "x"), score) for score in (MIN_SCORE + 0.05, MIN_SCORE - 0.2)]
    assert filter_hits(hits) == hits
