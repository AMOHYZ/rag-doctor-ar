from __future__ import annotations

from evals.report import (
    END,
    SPLIT_NAMES,
    START,
    Summary,
    headline_sentence,
    headline_table,
    inject,
    stage_table,
)


def _summary(hit: float, correct: float) -> Summary:
    row = {
        "n": 20,
        "hit_rate": hit,
        "correctness": correct,
        "groundedness": 0.9,
        "answered": 0.8,
        "tokens_per_query": 1500.4,
        "latency_p50_s": 2.04,
        "latency_p95_s": 6.5,
    }
    return {split: dict(row) for split in SPLIT_NAMES}


def test_headline_table_shows_before_and_after() -> None:
    table = headline_table(_summary(0.4, 0.3), _summary(0.9, 0.8), top_k=5)
    assert "| Gulf dialect (20) | 40% → 90% | 30% → 80% | 90% → 90% | 1500 → 1500 |" in table
    assert len(table.splitlines()) == 2 + len(SPLIT_NAMES)


def test_stage_table_has_a_row_per_stage_and_split() -> None:
    table = stage_table([("A", _summary(0.4, 0.3)), ("B", _summary(0.9, 0.8))], top_k=5)
    assert len(table.splitlines()) == 2 + 2 * len(SPLIT_NAMES)
    assert "| B | MSA | 90% | 80% | 90% | 80% | 1500 | 2.0 | 6.5 |" in table


def test_inject_replaces_only_the_marked_block() -> None:
    readme = f"intro\n{START}\nold\n{END}\noutro\n"
    assert inject(readme, "new") == f"intro\n{START}\nnew\n{END}\noutro\n"
    assert inject("no markers", "new") == "no markers"


def test_headline_sentence_leads_with_overall_correctness() -> None:
    sentence = headline_sentence(_summary(0.4, 0.3), _summary(0.9, 0.8))
    assert sentence.startswith("**Answer correctness: 30% → 80% overall**")
    assert "Gulf dialect 30% → 80%" in sentence
