"""Pure scoring: per-question records in, per-split summary and markdown out."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

SPLITS = ("msa", "gulf", "en")


@dataclass(frozen=True)
class Record:
    id: str
    lang: str
    question: str
    gold_answer: str
    gold_doc_ids: tuple[str, ...]
    answer: str
    retrieved_doc_ids: tuple[str, ...]
    scores: tuple[float, ...]
    hit: bool
    correctness: str  # correct | incorrect | judge_error
    groundedness: str  # supported | unsupported | no_answer | judge_error
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@dataclass(frozen=True)
class SplitSummary:
    split: str
    n: int
    hit_rate: float
    correctness: float
    groundedness: float  # supported / answers that made a claim
    answered: float  # share of questions where the system made a claim
    tokens_per_query: float
    latency_p50_s: float
    latency_p95_s: float


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize_split(split: str, records: Sequence[Record]) -> SplitSummary:
    n = len(records)
    if n == 0:
        return SplitSummary(split, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    claimed = [r for r in records if r.groundedness in ("supported", "unsupported")]
    supported = sum(r.groundedness == "supported" for r in claimed)
    latencies = [r.latency_s for r in records]
    return SplitSummary(
        split=split,
        n=n,
        hit_rate=sum(r.hit for r in records) / n,
        correctness=sum(r.correctness == "correct" for r in records) / n,
        groundedness=supported / len(claimed) if claimed else 0.0,
        answered=len(claimed) / n,
        tokens_per_query=sum(r.prompt_tokens + r.completion_tokens for r in records) / n,
        latency_p50_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, 95),
    )


def summarize(records: Sequence[Record]) -> list[SplitSummary]:
    rows = [summarize_split(split, [r for r in records if r.lang == split]) for split in SPLITS]
    return [*rows, summarize_split("overall", records)]


def to_markdown(rows: Sequence[SplitSummary], top_k: int) -> str:
    lines = [
        f"| Split | n | Hit@{top_k} | Correct | Grounded | Answered | Tokens/q | p50 s | p95 s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.split} | {row.n} | {row.hit_rate:.0%} | {row.correctness:.0%} "
            f"| {row.groundedness:.0%} | {row.answered:.0%} | {row.tokens_per_query:.0f} "
            f"| {row.latency_p50_s:.1f} | {row.latency_p95_s:.1f} |"
        )
    return "\n".join(lines)
