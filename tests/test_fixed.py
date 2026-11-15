from __future__ import annotations

from rag.baseline import MIN_SCORE
from rag.common import Chunk, Doc, Hit
from rag.fixed import CHUNK_CHARS, chunk_document, filter_hits, normalize_arabic, split_sentences

ENGLISH = " ".join(f"Sentence number {n} talks about the city of Dubai." for n in range(60))
ARABIC = " ".join(f"الجملة رقم {n} تتحدث عن مدينة الرياض وتاريخها." for n in range(60))


def test_filter_keeps_low_scoring_hits_instead_of_dropping_them() -> None:
    hits = [Hit(Chunk("d", "ar", "x"), score) for score in (MIN_SCORE + 0.05, MIN_SCORE - 0.2)]
    assert filter_hits(hits) == hits


def test_arabic_is_split_into_normal_sized_chunks() -> None:
    chunks = chunk_document(Doc("ar-1", "ar", "الرياض", ARABIC))
    assert len(chunks) > 3
    assert all(len(chunk.text) <= CHUNK_CHARS + len("الرياض\n") for chunk in chunks)
    assert all(chunk.text.startswith("الرياض\n") and chunk.text.endswith(".") for chunk in chunks)


def test_english_chunking_is_unchanged_in_shape() -> None:
    chunks = chunk_document(Doc("en-1", "en", "Dubai", ENGLISH))
    assert len(chunks) > 3
    assert all(chunk.text.endswith(".") for chunk in chunks)


def test_sentences_end_at_arabic_punctuation_and_paragraph_breaks() -> None:
    text = "ما عاصمة السعودية؟ الرياض هي العاصمة؛ وهي أكبر المدن.\n\nفقرة جديدة"
    assert split_sentences(text) == [
        "ما عاصمة السعودية؟",
        "الرياض هي العاصمة؛",
        "وهي أكبر المدن.",
        "فقرة جديدة",
    ]


def test_abbreviation_followed_by_lowercase_is_not_a_sentence_break() -> None:
    assert split_sentences("It is approx. five km away. Next one.") == [
        "It is approx. five km away.",
        "Next one.",
    ]


def test_run_on_sentence_is_split_below_the_chunk_limit() -> None:
    run_on = "، ".join(["جملة طويلة جدا بلا نقطة في آخرها"] * 80)
    pieces = split_sentences(run_on)
    assert len(pieces) > 1
    assert all(len(piece) <= CHUNK_CHARS for piece in pieces)


def test_normalize_arabic_folds_variants_diacritics_and_tatweel() -> None:
    assert normalize_arabic("الرِّيَاض") == "الرياض"
    assert normalize_arabic("جـــدة") == "جده"
    assert normalize_arabic("أبها إلى آل مستشفى") == "ابها الي ال مستشفي"
    assert normalize_arabic("Dubai 2024") == "Dubai 2024"
