"""Baseline pipeline. It runs end to end and looks reasonable, but it has two planted bugs.

BUG 1 (post-retrieval filter): `MIN_SCORE` is an absolute cosine cutoff that was tuned by eye
on a few English queries. Scores for Arabic, and especially dialect, queries sit lower, so the
filter silently throws away relevant chunks, sometimes all of them.

BUG 2 (English-only chunking): sentences are split on ASCII punctuation followed by a capital
letter. Arabic has no capitals, so an Arabic article is never split and becomes one huge chunk;
the tail is clipped when the prompt is built. Nothing is normalised either (alef/ya/ta-marbuta
variants, diacritics, tatweel), so informal spellings do not line up with the indexed text.
"""

from __future__ import annotations

import re

from rag.common import Chunk, Doc, Hit, PipelineSpec

CHUNK_CHARS = 700
MIN_SCORE = 0.55  # BUG 1
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")  # BUG 2


def split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text).strip()
    return [sentence for sentence in _SENTENCE_BREAK.split(flat) if sentence]


def chunk_document(doc: Doc) -> list[Chunk]:
    """Pack whole sentences into chunks of about CHUNK_CHARS; a sentence is never broken."""
    bodies: list[str] = []
    current = ""
    for sentence in split_sentences(doc.text):
        if current and len(current) + len(sentence) + 1 > CHUNK_CHARS:
            bodies.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        bodies.append(current)
    return [Chunk(doc.id, doc.lang, f"{doc.title}\n{body}") for body in bodies]


def prepare_text(text: str) -> str:
    return text


def filter_hits(hits: list[Hit]) -> list[Hit]:
    return [hit for hit in hits if hit.score >= MIN_SCORE]


SPEC = PipelineSpec(
    name="baseline",
    chunker=chunk_document,
    prepare_text=prepare_text,
    post_filter=filter_hits,
    fetch_k=10,
    top_k=5,
)
