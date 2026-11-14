"""Shared building blocks: data types, embeddings, vector store, LLM clients, prompt.

The baseline and fixed pipelines differ only in the three functions they plug into
`PipelineSpec` (chunker, text preparation, post-retrieval filter). Everything else is here
and in rag/graph.py, which wires the steps into a LangGraph graph.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import faiss
import httpx
import numpy as np
from numpy.typing import NDArray

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_LLM_MODELS = {
    "ollama": "qwen3:8b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}
MAX_CHUNK_CHARS = 1200  # per-chunk cap in the prompt, keeps prompt size (and cost) bounded
NO_CONTEXT_ANSWER = "I don't know. / لا أعرف."
_TIMEOUT = httpx.Timeout(300.0)


@dataclass(frozen=True)
class Doc:
    id: str
    lang: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    lang: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class Answer:
    text: str
    hits: tuple[Hit, ...]
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class LLM(Protocol):
    def complete(self, prompt: str, *, json_mode: bool = False) -> Completion: ...


@dataclass(frozen=True)
class PipelineSpec:
    """The parts that differ between the broken and the fixed pipeline."""

    name: str
    chunker: Callable[[Doc], list[Chunk]]
    prepare_text: Callable[[str], str]  # applied to chunks and queries before embedding
    post_filter: Callable[[list[Hit]], list[Hit]]
    fetch_k: int
    top_k: int


# --------------------------------------------------------------------------- corpus


def load_corpus(path: Path) -> list[Doc]:
    docs: list[Doc] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                docs.append(
                    Doc(id=row["id"], lang=row["lang"], title=row["title"], text=row["text"])
                )
    return docs


# ----------------------------------------------------------------------- embeddings


class OllamaEmbedder:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or os.environ.get("EMBED_MODEL") or DEFAULT_EMBED_MODEL
        self._host = host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST

    def __repr__(self) -> str:  # stable across runs, so lgcov node fingerprints stay stable
        return f"OllamaEmbedder({self.model!r})"

    def embed(self, texts: Sequence[str], batch_size: int = 16) -> NDArray[np.float32]:
        rows: list[list[float]] = []
        with httpx.Client(timeout=_TIMEOUT) as client:
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                response = client.post(
                    f"{self._host}/api/embed", json={"model": self.model, "input": batch}
                )
                response.raise_for_status()
                rows.extend(response.json()["embeddings"])
        return np.asarray(rows, dtype=np.float32)


def cached_embed(embedder: Embedder, texts: Sequence[str], cache_file: Path) -> NDArray[np.float32]:
    """Embed `texts`, reusing the on-disk vectors when model and texts are unchanged."""
    digest = hashlib.sha256("\x00".join([embedder.model, *texts]).encode("utf-8")).hexdigest()
    if cache_file.exists():
        with np.load(cache_file) as cached:
            if str(cached["digest"]) == digest:
                return np.asarray(cached["vectors"], dtype=np.float32)
    vectors = embedder.embed(texts)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, digest=np.array(digest), vectors=vectors)
    return vectors


# --------------------------------------------------------------------- vector store


class VectorStore:
    """Exact cosine-similarity search over chunk embeddings (FAISS inner product)."""

    def __init__(self, chunks: Sequence[Chunk], vectors: NDArray[np.float32]) -> None:
        self._chunks = list(chunks)
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(_unit(vectors))

    def search(self, query_vector: NDArray[np.float32], k: int) -> list[Hit]:
        scores, ids = self._index.search(_unit(query_vector.reshape(1, -1)), k)
        return [
            Hit(self._chunks[int(i)], float(s))
            for s, i in zip(scores[0], ids[0], strict=True)
            if i >= 0
        ]


def _unit(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.asarray(vectors / np.maximum(norms, 1e-12), dtype=np.float32)


# ---------------------------------------------------------------------- LLM clients


class OllamaLLM:
    def __init__(self, model: str, host: str | None = None, think: bool = False) -> None:
        self._model = model
        self._think = think  # let reasoning models think first: slower, more reliable (judging)
        self._host = host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST

    def __repr__(self) -> str:
        return f"OllamaLLM({self._model!r})"

    def complete(self, prompt: str, *, json_mode: bool = False) -> Completion:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": self._think,
            "options": {
                "temperature": 0,
                "seed": 7,
                "num_ctx": 8192,
                "num_predict": 2000 if self._think else 300,
            },
        }
        if json_mode:
            payload["format"] = "json"
        response = httpx.post(f"{self._host}/api/chat", json=payload, timeout=_TIMEOUT)
        response.raise_for_status()
        body = response.json()
        return Completion(
            text=str(body["message"]["content"]).strip(),
            prompt_tokens=int(body.get("prompt_eval_count", 0)),
            completion_tokens=int(body.get("eval_count", 0)),
        )


class OpenAILLM:
    def __init__(self, model: str, api_key: str) -> None:
        self._model = model
        self._api_key = api_key

    def __repr__(self) -> str:
        return f"OpenAILLM({self._model!r})"

    def complete(self, prompt: str, *, json_mode: bool = False) -> Completion:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        body = response.json()
        return Completion(
            text=str(body["choices"][0]["message"]["content"]).strip(),
            prompt_tokens=int(body["usage"]["prompt_tokens"]),
            completion_tokens=int(body["usage"]["completion_tokens"]),
        )


class AnthropicLLM:
    def __init__(self, model: str, api_key: str) -> None:
        self._model = model
        self._api_key = api_key

    def __repr__(self) -> str:
        return f"AnthropicLLM({self._model!r})"

    def complete(self, prompt: str, *, json_mode: bool = False) -> Completion:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            json={
                "model": self._model,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={"x-api-key": self._api_key, "anthropic-version": "2023-06-01"},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        body = response.json()
        text = "".join(block.get("text", "") for block in body["content"])
        return Completion(
            text=text.strip(),
            prompt_tokens=int(body["usage"]["input_tokens"]),
            completion_tokens=int(body["usage"]["output_tokens"]),
        )


def make_llm(provider: str | None = None, model: str | None = None, think: bool = False) -> LLM:
    """Build an LLM client from arguments or LLM_PROVIDER / LLM_MODEL. Keys come from env only."""
    provider = (provider or os.environ.get("LLM_PROVIDER") or "ollama").lower()
    if provider not in DEFAULT_LLM_MODELS:
        raise ValueError(
            f"LLM_PROVIDER must be one of {sorted(DEFAULT_LLM_MODELS)}, got {provider!r}"
        )
    model = model or os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODELS[provider]
    if provider == "ollama":
        return OllamaLLM(model, think=think)
    key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    api_key = os.environ.get(key_name)
    if not api_key:
        raise RuntimeError(f"{key_name} is not set (required for LLM_PROVIDER={provider})")
    return OpenAILLM(model, api_key) if provider == "openai" else AnthropicLLM(model, api_key)


# --------------------------------------------------------------------------- prompt


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    passages = "\n\n".join(
        f"[{n}] {hit.chunk.text[:MAX_CHUNK_CHARS]}" for n, hit in enumerate(hits, start=1)
    )
    return (
        "Answer the question using only the passages below. "
        "Reply in the same language as the question, in one or two short sentences. "
        "If the passages do not contain the answer, say you don't know.\n\n"
        f"Passages:\n{passages}\n\nQuestion: {question}\nAnswer:"
    )
