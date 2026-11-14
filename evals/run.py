"""Run a pipeline over evals/questions.jsonl and write a markdown + JSON report.

python -m evals.run --pipeline baseline --out reports/baseline
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from evals.judge import judge_correctness, judge_groundedness
from evals.metrics import Record, summarize, to_markdown
from rag.common import (
    DEFAULT_LLM_MODELS,
    LLM,
    OllamaEmbedder,
    PipelineSpec,
    load_corpus,
    make_llm,
)
from rag.graph import Index, RagGraph, answer_question, build_graph

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "evals" / "questions.jsonl"
CORPUS = ROOT / "data" / "corpus.jsonl"


def load_spec(name: str) -> PipelineSpec:
    spec = getattr(importlib.import_module(f"rag.{name}"), "SPEC", None)
    if not isinstance(spec, PipelineSpec):
        raise ValueError(f"rag.{name} does not define a PipelineSpec named SPEC")
    return spec


def load_questions(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate_question(graph: RagGraph, judge: LLM, item: dict[str, object]) -> Record:
    question = str(item["question"])
    gold_answer = str(item["gold_answer"])
    gold_doc_ids = tuple(str(doc_id) for doc_id in item["gold_doc_ids"])  # type: ignore[attr-defined]  # JSON list
    answer = answer_question(graph, question)
    retrieved = tuple(hit.chunk.doc_id for hit in answer.hits)
    return Record(
        id=str(item["id"]),
        lang=str(item["lang"]),
        question=question,
        gold_answer=gold_answer,
        gold_doc_ids=gold_doc_ids,
        answer=answer.text,
        retrieved_doc_ids=retrieved,
        scores=tuple(round(hit.score, 4) for hit in answer.hits),
        hit=any(doc_id in retrieved for doc_id in gold_doc_ids),
        correctness=judge_correctness(judge, question, gold_answer, answer.text),
        groundedness=judge_groundedness(judge, question, answer.text, answer.hits),
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        latency_s=round(answer.latency_s, 3),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True, choices=["baseline", "fixed"])
    parser.add_argument("--out", required=True, type=Path, help="output path without extension")
    parser.add_argument("--label", default=None, help="title used in the report")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    args = parser.parse_args()

    spec = load_spec(args.pipeline)
    embedder = OllamaEmbedder()
    judge = make_llm(
        os.environ.get("JUDGE_PROVIDER") or None,
        os.environ.get("JUDGE_MODEL") or None,
        think=True,  # the judge reasons before its verdict; without this a small model misgrades
    )
    index = Index(spec.name, embedder, cache_dir=ROOT / ".cache")
    n_chunks = index.build(load_corpus(CORPUS), spec.chunker, spec.prepare_text)
    graph = build_graph(spec, index, make_llm())
    questions = load_questions(QUESTIONS)[: args.limit]
    print(f"{spec.name}: {n_chunks} chunks indexed, {len(questions)} questions", flush=True)

    records: list[Record] = []
    for n, item in enumerate(questions, start=1):
        record = evaluate_question(graph, judge, item)
        records.append(record)
        print(
            f"[{n}/{len(questions)}] {record.id} hit={int(record.hit)} "
            f"{record.correctness} {record.groundedness} {record.latency_s:.1f}s",
            flush=True,
        )

    rows = summarize(records)
    label = args.label or spec.name
    provider = (os.environ.get("LLM_PROVIDER") or "ollama").lower()
    meta = {
        "label": label,
        "pipeline": spec.name,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "llm": f"{provider}/{os.environ.get('LLM_MODEL') or DEFAULT_LLM_MODELS[provider]}",
        "embeddings": embedder.model,
        "top_k": spec.top_k,
        "chunks_indexed": n_chunks,
        "questions": len(records),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "summary": [dataclasses.asdict(row) for row in rows],
        "records": [dataclasses.asdict(record) for record in records],
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    table = to_markdown(rows, spec.top_k)
    args.out.with_suffix(".md").write_text(
        f"# Eval report: {label}\n\n"
        f"{meta['questions']} questions, {n_chunks} chunks, LLM `{meta['llm']}`, "
        f"embeddings `{meta['embeddings']}`, generated {meta['generated_at']}.\n\n{table}\n\n"
        "Grounded = supported answers / answers that made a claim. Answered = share of questions "
        "where the system made a claim instead of abstaining. Tokens and latency cover retrieval "
        "and generation only (not the judge).\n",
        encoding="utf-8",
    )
    print(table)


if __name__ == "__main__":
    main()
