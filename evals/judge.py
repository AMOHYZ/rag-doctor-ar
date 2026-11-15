"""LLM judges for answer correctness and groundedness. Prompts live in evals/prompts/."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from rag.common import LLM, MAX_CHUNK_CHARS, NO_CONTEXT_ANSWER, Hit

_PROMPTS = Path(__file__).parent / "prompts"
CORRECTNESS_PROMPT = (_PROMPTS / "correctness.txt").read_text(encoding="utf-8")
GROUNDEDNESS_PROMPT = (_PROMPTS / "groundedness.txt").read_text(encoding="utf-8")
JUDGE_ERROR = "judge_error"


def parse_verdict(raw: str, allowed: Sequence[str]) -> str:
    """Pull the verdict out of the judge's JSON reply; anything unparseable is a judge error."""
    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if match:
        try:
            verdict = json.loads(match.group(0)).get("verdict")
        except (json.JSONDecodeError, AttributeError):
            verdict = None
        if verdict in allowed:
            return str(verdict)
    return JUDGE_ERROR


def judge_correctness(llm: LLM, question: str, gold_answer: str, answer: str) -> str:
    if answer == NO_CONTEXT_ANSWER:  # the pipeline's canned abstention is never correct
        return "incorrect"
    prompt = CORRECTNESS_PROMPT.format(question=question, gold_answer=gold_answer, answer=answer)
    return parse_verdict(llm.complete(prompt, json_mode=True).text, ("correct", "incorrect"))


def judge_groundedness(llm: LLM, question: str, answer: str, hits: Sequence[Hit]) -> str:
    """The judge sees exactly the passages the generator saw (same per-chunk clipping)."""
    if not hits:
        return "no_answer"
    passages = "\n\n".join(
        f"[{n}] {hit.chunk.text[:MAX_CHUNK_CHARS]}" for n, hit in enumerate(hits, start=1)
    )
    prompt = GROUNDEDNESS_PROMPT.format(passages=passages, question=question, answer=answer)
    return parse_verdict(
        llm.complete(prompt, json_mode=True).text, ("supported", "unsupported", "no_answer")
    )
