"""Fixed pipeline: the same shape as rag/baseline.py with both planted bugs repaired.

FIX 1 (post-retrieval filter): no absolute score cutoff. Cosine scores are not comparable
across languages or query styles, so we rank and keep the top-k instead of thresholding.

FIX 2 (Arabic-aware chunking): sentences also end at Arabic punctuation and paragraph breaks,
no capital letter is required, and over-long sentences are split, so Arabic articles become
normal-sized chunks. Text is normalised before embedding, for both chunks and queries
(alef/ya/ta-marbuta variants, diacritics, tatweel).
"""

from __future__ import annotations

import re

from rag.common import Chunk, Doc, Hit, PipelineSpec

CHUNK_CHARS = 700
# End of sentence: ASCII or Arabic punctuation, unless a lowercase Latin letter follows
# (keeps "approx. five km" together). Paragraph breaks always end a sentence.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?؟؛])\s+(?![a-z])|\n+")
_CLAUSE_BREAK = re.compile(r"(?<=[،,;:])\s+")
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_LETTER_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})


def normalize_arabic(text: str) -> str:
    """Fold spelling variants that users mix freely; a no-op for non-Arabic text."""
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    return text.translate(_LETTER_VARIANTS)


def _pack(pieces: list[str], limit: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > limit:
            packed.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        packed.append(current)
    return packed


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for sentence in _SENTENCE_BREAK.split(text):
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if len(sentence) > CHUNK_CHARS:  # run-on sentence: fall back to clauses, then words
            clauses = _pack(_CLAUSE_BREAK.split(sentence), CHUNK_CHARS)
            for clause in clauses:
                sentences.extend(_pack(clause.split(" "), CHUNK_CHARS))
        elif sentence:
            sentences.append(sentence)
    return sentences


def chunk_document(doc: Doc) -> list[Chunk]:
    """Pack whole sentences into chunks of about CHUNK_CHARS, for any script."""
    bodies = _pack(split_sentences(doc.text), CHUNK_CHARS)
    return [Chunk(doc.id, doc.lang, f"{doc.title}\n{body}") for body in bodies]


def prepare_text(text: str) -> str:
    return normalize_arabic(text)


def filter_hits(hits: list[Hit]) -> list[Hit]:
    return hits


SPEC = PipelineSpec(
    name="fixed",
    chunker=chunk_document,
    prepare_text=prepare_text,
    post_filter=filter_hits,
    fetch_k=10,
    top_k=5,
)
