"""These tests pin down the planted bugs so the baseline stays broken on purpose."""

from __future__ import annotations

from rag.baseline import CHUNK_CHARS, MIN_SCORE, chunk_document, filter_hits
from rag.common import Chunk, Doc, Hit

ENGLISH = " ".join(f"Sentence number {n} talks about the city of Dubai." for n in range(60))
ARABIC = " ".join(f"الجملة رقم {n} تتحدث عن مدينة الرياض وتاريخها." for n in range(60))


def test_english_is_split_into_sentence_aligned_chunks() -> None:
    chunks = chunk_document(Doc("en-1", "en", "Dubai", ENGLISH))
    assert len(chunks) > 3
    assert all(len(chunk.text) <= CHUNK_CHARS + len("Dubai\n") for chunk in chunks)
    assert all(chunk.text.startswith("Dubai\n") and chunk.text.endswith(".") for chunk in chunks)


def test_planted_bug_arabic_is_never_split() -> None:
    chunks = chunk_document(Doc("ar-1", "ar", "الرياض", ARABIC))
    assert len(chunks) == 1
    assert len(chunks[0].text) > 3 * CHUNK_CHARS


def test_planted_bug_filter_drops_everything_below_absolute_cutoff() -> None:
    hits = [Hit(Chunk("d", "ar", "x"), score) for score in (MIN_SCORE + 0.05, MIN_SCORE - 0.01)]
    assert filter_hits(hits) == hits[:1]
    assert filter_hits(hits[1:]) == []
