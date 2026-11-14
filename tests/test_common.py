from __future__ import annotations

from pathlib import Path

import pytest

from rag.common import (
    MAX_CHUNK_CHARS,
    Chunk,
    Doc,
    Hit,
    PipelineSpec,
    VectorStore,
    build_prompt,
    load_corpus,
    make_llm,
)
from tests.fakes import FakeEmbedder

DOCS = [
    Doc(id="en-1", lang="en", title="Riyadh", text="Riyadh is the capital of Saudi Arabia."),
    Doc(id="en-2", lang="en", title="Falafel", text="Falafel is a fried chickpea ball."),
    Doc(id="ar-1", lang="ar", title="جدة", text="جدة مدينة ساحلية على البحر الأحمر."),
]


def _one_chunk_per_doc(doc: Doc) -> list[Chunk]:
    return [Chunk(doc_id=doc.id, lang=doc.lang, text=doc.text)]


def _spec(top_k: int = 2) -> PipelineSpec:
    return PipelineSpec(
        name="test",
        chunker=_one_chunk_per_doc,
        prepare_text=lambda text: text,
        post_filter=lambda hits: hits,
        fetch_k=3,
        top_k=top_k,
    )


def test_vector_store_ranks_matching_chunk_first() -> None:
    embedder = FakeEmbedder()
    chunks = [c for d in DOCS for c in _one_chunk_per_doc(d)]
    store = VectorStore(chunks, embedder.embed([c.text for c in chunks]))
    hits = store.search(embedder.embed(["fried chickpea ball"])[0], k=2)
    assert hits[0].chunk.doc_id == "en-2"
    assert hits[0].score >= hits[1].score


def test_build_prompt_contains_question_and_clips_long_chunks() -> None:
    long_hit = Hit(Chunk("d", "en", "x" * (MAX_CHUNK_CHARS + 500)), 0.9)
    prompt = build_prompt("What is x?", [long_hit])
    assert "What is x?" in prompt
    assert "x" * MAX_CHUNK_CHARS in prompt
    assert "x" * (MAX_CHUNK_CHARS + 1) not in prompt


def test_load_corpus_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"id": "ar-1", "lang": "ar", "title": "جدة", "text": "نص", "url": "u"}\n',
        encoding="utf-8",
    )
    assert load_corpus(path) == [Doc(id="ar-1", lang="ar", title="جدة", text="نص")]


def test_make_llm_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        make_llm(provider="nope")


def test_hosted_providers_require_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_llm(provider="openai")
