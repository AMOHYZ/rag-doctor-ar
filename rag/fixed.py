"""Fixed pipeline: the same shape as rag/baseline.py with the planted bugs repaired.

FIX 1 (post-retrieval filter): no absolute score cutoff. Cosine scores are not comparable
across languages or query styles, so we rank and keep the top-k instead of thresholding.

Still open: the English-only chunking (bug 2) is reused from the baseline for now.
"""

from __future__ import annotations

from rag.baseline import chunk_document, prepare_text
from rag.common import Hit, PipelineSpec


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
