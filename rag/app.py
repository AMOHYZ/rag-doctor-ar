"""Module-level graph for tools that load a LangGraph graph by path, such as lgcov.

    lgcov.yaml -> graph_entrypoint: rag/app.py:graph

Importing this touches neither the network nor the corpus: the index is built on the first
search, so replaying recorded runs works on a machine without Ollama.
"""

from __future__ import annotations

from pathlib import Path

from rag.common import OllamaEmbedder, make_llm
from rag.fixed import SPEC
from rag.graph import Index, build_graph

ROOT = Path(__file__).resolve().parent.parent
graph = build_graph(
    SPEC,
    Index(SPEC.name, OllamaEmbedder(), ROOT / ".cache", corpus=ROOT / "data" / "corpus.jsonl"),
    make_llm(),
)
