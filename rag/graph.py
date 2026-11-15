"""The RAG pipeline as a LangGraph graph: retrieve -> post_filter -> generate.

Each step is its own node so a regression can be traced to the node that caused it
(see the regression-guard section of the README). State holds plain dicts, so runs can be
recorded and replayed as JSON.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag.common import (
    LLM,
    NO_CONTEXT_ANSWER,
    Answer,
    Chunk,
    Doc,
    Embedder,
    Hit,
    PipelineSpec,
    VectorStore,
    build_prompt,
    cached_embed,
    load_corpus,
)


class HitDict(TypedDict):
    doc_id: str
    lang: str
    text: str
    score: float


class RagState(TypedDict, total=False):
    question: str
    candidates: list[HitDict]  # raw vector-search results
    hits: list[HitDict]  # what survives the post-retrieval filter, top-k
    answer: str
    prompt_tokens: int
    completion_tokens: int


RagGraph = CompiledStateGraph[RagState, None, RagState, RagState]
Chunker = Callable[[Doc], list[Chunk]]
TextPrep = Callable[[str], str]


def _to_dicts(hits: Sequence[Hit]) -> list[HitDict]:
    return [
        HitDict(doc_id=h.chunk.doc_id, lang=h.chunk.lang, text=h.chunk.text, score=h.score)
        for h in hits
    ]


def _to_hits(rows: Sequence[HitDict]) -> list[Hit]:
    return [Hit(Chunk(r["doc_id"], r["lang"], r["text"]), r["score"]) for r in rows]


class Index:
    """Chunk embeddings for one pipeline, built on first use and cached on disk."""

    def __init__(
        self, name: str, embedder: Embedder, cache_dir: Path, corpus: Path | None = None
    ) -> None:
        self._name = name
        self._embedder = embedder
        self._cache_file = cache_dir / f"index-{name}.npz"
        self._corpus = corpus  # if set, the index builds itself from this file on first use
        self._store: VectorStore | None = None

    def __repr__(self) -> str:  # stable across runs, so lgcov node fingerprints stay stable
        return f"Index({self._name!r})"

    def build(self, docs: Sequence[Doc], chunker: Chunker, prepare_text: TextPrep) -> int:
        chunks = [chunk for doc in docs for chunk in chunker(doc)]
        texts = [prepare_text(chunk.text) for chunk in chunks]
        self._store = VectorStore(chunks, cached_embed(self._embedder, texts, self._cache_file))
        return len(chunks)

    def ensure_built(self, chunker: Chunker, prepare_text: TextPrep) -> None:
        if self._store is None and self._corpus is not None:
            self.build(load_corpus(self._corpus), chunker, prepare_text)

    def search(self, prepared_query: str, k: int) -> list[Hit]:
        if self._store is None:
            raise RuntimeError("the index is empty: call Index.build() or pass a corpus path")
        return self._store.search(self._embedder.embed([prepared_query])[0], k)


def build_graph(spec: PipelineSpec, index: Index, llm: LLM) -> RagGraph:
    # Each node closes over only the functions it uses. Tools that fingerprint node code
    # (lgcov) can then tell which node a code change touches: the chunker and text
    # preparation belong to `retrieve`, the filter to `post_filter`, the prompt to `generate`.
    chunker, prepare_text, keep = spec.chunker, spec.prepare_text, spec.post_filter
    fetch_k, top_k = spec.fetch_k, spec.top_k

    def retrieve(state: RagState) -> RagState:
        index.ensure_built(chunker, prepare_text)
        return {"candidates": _to_dicts(index.search(prepare_text(state["question"]), fetch_k))}

    def post_filter(state: RagState) -> RagState:
        return {"hits": _to_dicts(keep(_to_hits(state["candidates"]))[:top_k])}

    def generate(state: RagState) -> RagState:
        hits = _to_hits(state["hits"])
        if not hits:
            return {"answer": NO_CONTEXT_ANSWER, "prompt_tokens": 0, "completion_tokens": 0}
        completion = llm.complete(build_prompt(state["question"], hits))
        return {
            "answer": completion.text,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
        }

    builder = StateGraph(RagState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("post_filter", post_filter)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "post_filter")
    builder.add_edge("post_filter", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


def answer_question(graph: RagGraph, question: str) -> Answer:
    started = time.perf_counter()
    state = graph.invoke({"question": question})
    return Answer(
        text=state["answer"],
        hits=tuple(_to_hits(state["hits"])),
        prompt_tokens=state["prompt_tokens"],
        completion_tokens=state["completion_tokens"],
        latency_s=time.perf_counter() - started,
    )
