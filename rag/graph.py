"""The RAG pipeline as a LangGraph graph: retrieve -> post_filter -> generate.

Each step is its own node so a regression can be traced to the node that caused it
(see the regression-guard section of the README). State holds plain dicts, so runs can be
recorded and replayed as JSON.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
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


def _to_dicts(hits: Sequence[Hit]) -> list[HitDict]:
    return [
        HitDict(doc_id=h.chunk.doc_id, lang=h.chunk.lang, text=h.chunk.text, score=h.score)
        for h in hits
    ]


def _to_hits(rows: Sequence[HitDict]) -> list[Hit]:
    return [Hit(Chunk(r["doc_id"], r["lang"], r["text"]), r["score"]) for r in rows]


class Index:
    """Chunks + embeddings for one pipeline spec, built on first use and cached on disk."""

    def __init__(self, spec: PipelineSpec, embedder: Embedder, cache_dir: Path) -> None:
        self._spec = spec
        self._embedder = embedder
        self._cache_file = cache_dir / f"index-{spec.name}.npz"
        self._store: VectorStore | None = None
        self.n_chunks = 0

    def __repr__(self) -> str:  # stable across runs, so lgcov node fingerprints stay stable
        return f"Index({self._spec.name!r})"

    def build(self, docs: Sequence[Doc]) -> int:
        chunks = [chunk for doc in docs for chunk in self._spec.chunker(doc)]
        texts = [self._spec.prepare_text(chunk.text) for chunk in chunks]
        self._store = VectorStore(chunks, cached_embed(self._embedder, texts, self._cache_file))
        self.n_chunks = len(chunks)
        return self.n_chunks

    def search(self, question: str, k: int) -> list[Hit]:
        if self._store is None:
            raise RuntimeError("call Index.build() before searching")
        query_vector = self._embedder.embed([self._spec.prepare_text(question)])[0]
        return self._store.search(query_vector, k)


def build_graph(spec: PipelineSpec, index: Index, llm: LLM) -> RagGraph:
    def retrieve(state: RagState) -> RagState:
        return {"candidates": _to_dicts(index.search(state["question"], spec.fetch_k))}

    def post_filter(state: RagState) -> RagState:
        kept = spec.post_filter(_to_hits(state["candidates"]))[: spec.top_k]
        return {"hits": _to_dicts(kept)}

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
