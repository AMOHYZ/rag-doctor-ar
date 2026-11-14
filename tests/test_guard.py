from __future__ import annotations

from evals.guard import Summary, Violation, check

REFERENCE: Summary = {
    "gulf": {"hit_rate": 0.80, "correctness": 0.70},
    "overall": {"hit_rate": 0.90, "correctness": 0.80},
}


def test_small_drop_within_threshold_passes() -> None:
    candidate: Summary = {
        "gulf": {"hit_rate": 0.75, "correctness": 0.70},
        "overall": {"hit_rate": 0.90, "correctness": 0.85},
    }
    assert check(REFERENCE, candidate, max_drop=0.05) == []


def test_drop_beyond_threshold_is_reported_per_split_and_metric() -> None:
    candidate: Summary = {
        "gulf": {"hit_rate": 0.40, "correctness": 0.70},
        "overall": {"hit_rate": 0.90, "correctness": 0.60},
    }
    assert check(REFERENCE, candidate, max_drop=0.05) == [
        Violation("gulf", "hit_rate", 0.80, 0.40),
        Violation("overall", "correctness", 0.80, 0.60),
    ]


def test_missing_split_in_candidate_counts_as_a_regression() -> None:
    assert len(check(REFERENCE, {"overall": REFERENCE["overall"]}, max_drop=0.05)) == 2
