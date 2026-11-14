"""Regression guard: fail when hit rate or answer correctness drops past a threshold.

    python -m evals.guard --reference reports/guard_reference.json --candidate reports/fixed.json

Exit code 1 on a regression, so it can gate a pull request in CI.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

GUARDED_METRICS = ("hit_rate", "correctness")
Summary = dict[str, dict[str, float]]  # split -> metric -> value


@dataclass(frozen=True)
class Violation:
    split: str
    metric: str
    reference: float
    candidate: float


def load_summary(path: Path) -> Summary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["split"]: row for row in payload["summary"]}


def check(reference: Summary, candidate: Summary, max_drop: float) -> list[Violation]:
    """Every (split, metric) where the candidate is more than `max_drop` below the reference."""
    violations: list[Violation] = []
    for split, ref_row in reference.items():
        for metric in GUARDED_METRICS:
            cand_value = candidate.get(split, {}).get(metric, 0.0)
            if ref_row[metric] - cand_value > max_drop + 1e-9:
                violations.append(Violation(split, metric, ref_row[metric], cand_value))
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--max-drop", type=float, default=0.05, help="allowed absolute drop")
    args = parser.parse_args()

    violations = check(load_summary(args.reference), load_summary(args.candidate), args.max_drop)
    if not violations:
        print(f"guard OK: no metric dropped more than {args.max_drop:.0%} vs {args.reference}")
        return
    print(f"guard FAILED: {args.candidate} regressed vs {args.reference}\n")
    print("| split | metric | reference | candidate | drop |")
    print("|---|---|---|---|---|")
    for v in violations:
        drop = v.reference - v.candidate
        print(f"| {v.split} | {v.metric} | {v.reference:.0%} | {v.candidate:.0%} | -{drop:.0%} |")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
