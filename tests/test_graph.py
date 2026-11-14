from __future__ import annotations

from pathlib import Path

from rag.common import Chunk, Doc, PipelineSpec
from rag.graph import Index, answer_question, build_graph
from tests.fakes import FakeEmbedder, FakeLLM

DOCS = [
    Doc(id="en-1", lang="en", title="Riyadh", text="Riyadh is the capital of Saudi Arabia."),
    Doc(id="en-2", lang="en", title="Falafel", text="Falafel is a fried chickpea ball."),
    Doc(id="ar-1", lang="ar", title="جدة", text="جدة مدينة ساحلية على البحر الأحمر."),
]


def _one_chunk_per_doc(doc: Doc) -> list[Chunk]:
    return [Chunk(doc_id=doc.id, lang=doc.lang, text=doc.text)]


def _spec(top_k: int = 2) -> PipelineSpec:
    return PipelineSpec("test", _one_chunk_per_doc, lambda t: t, lambda hits: hits, 3, top_k)


def test_graph_has_one_node_per_pipeline_step(tmp_path: Path) -> None:
    graph = build_graph(_spec(), Index("test", FakeEmbedder(), tmp_path), FakeLLM())
    assert {"retrieve", "post_filter", "generate"} <= set(graph.get_graph().nodes)


def test_graph_answers_from_retrieved_context(tmp_path: Path) -> None:
    llm = FakeLLM("Riyadh")
    index = Index("test", FakeEmbedder(), tmp_path)
    assert index.build(DOCS, _one_chunk_per_doc, str) == 3
    answer = answer_question(
        build_graph(_spec(), index, llm), "What is the capital of Saudi Arabia?"
    )
    assert answer.text == "Riyadh"
    assert answer.hits[0].chunk.doc_id == "en-1"
    assert len(answer.hits) == 2
    assert answer.prompt_tokens > 0


def test_graph_does_not_call_llm_when_filter_drops_everything(tmp_path: Path) -> None:
    llm = FakeLLM()
    spec = PipelineSpec("empty", _one_chunk_per_doc, lambda t: t, lambda hits: [], 3, 2)
    index = Index("empty", FakeEmbedder(), tmp_path)
    index.build(DOCS, _one_chunk_per_doc, str)
    answer = answer_question(build_graph(spec, index, llm), "anything")
    assert answer.hits == ()
    assert llm.prompts == []
    assert answer.prompt_tokens == 0


def test_index_embeddings_are_cached_on_disk(tmp_path: Path) -> None:
    calls = 0

    class CountingEmbedder(FakeEmbedder):
        def embed(self, texts):  # type: ignore[no-untyped-def]  # mirrors FakeEmbedder.embed
            nonlocal calls
            calls += 1
            return super().embed(texts)

    for _ in range(2):
        Index("test", CountingEmbedder(), tmp_path).build(DOCS, _one_chunk_per_doc, str)
    assert calls == 1


def test_graph_builds_the_index_from_a_corpus_file_on_first_use(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id": "en-2", "lang": "en", "title": "Falafel", "text": "A fried chickpea ball."}\n',
        encoding="utf-8",
    )
    index = Index("test", FakeEmbedder(), tmp_path, corpus=corpus)
    answer = answer_question(build_graph(_spec(), index, FakeLLM()), "chickpea")
    assert answer.hits[0].chunk.doc_id == "en-2"
