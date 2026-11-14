"""`make report`: combine the per-stage eval reports into a before/after markdown summary.

Also refreshes the results block in README.md, so README numbers always come from real runs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
README = ROOT / "README.md"
STAGES = (
    ("baseline", "Baseline (broken)"),
    ("fix1_filter", "Fix 1: filter"),
    ("fixed", "Fix 2: + Arabic chunking"),
)
SPLIT_NAMES = {"msa": "MSA", "gulf": "Gulf dialect", "en": "English", "overall": "**Overall**"}
START, END = "<!-- results:start -->", "<!-- results:end -->"

Summary = dict[str, dict[str, float]]  # split -> metric -> value


def load_summary(path: Path) -> Summary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["split"]: row for row in payload["summary"]}


def headline_table(before: Summary, after: Summary, top_k: int) -> str:
    lines = [
        f"| Questions | Hit@{top_k} | Answer correct | Grounded | Tokens/query | p95 latency |",
        "|---|---|---|---|---|---|",
    ]
    for split, name in SPLIT_NAMES.items():
        b, a = before[split], after[split]
        lines.append(
            f"| {name} ({int(b['n'])}) "
            f"| {b['hit_rate']:.0%} → {a['hit_rate']:.0%} "
            f"| {b['correctness']:.0%} → {a['correctness']:.0%} "
            f"| {b['groundedness']:.0%} → {a['groundedness']:.0%} "
            f"| {b['tokens_per_query']:.0f} → {a['tokens_per_query']:.0f} "
            f"| {b['latency_p95_s']:.1f}s → {a['latency_p95_s']:.1f}s |"
        )
    return "\n".join(lines)


def headline_sentence(before: Summary, after: Summary) -> str:
    def change(split: str) -> str:
        return f"{before[split]['correctness']:.0%} → {after[split]['correctness']:.0%}"

    return (
        f"**Answer correctness: {change('overall')} overall** "
        f"(MSA {change('msa')}, Gulf dialect {change('gulf')}, English {change('en')})."
    )


def stage_table(stages: Sequence[tuple[str, Summary]], top_k: int) -> str:
    lines = [
        f"| Stage | Split | Hit@{top_k} | Correct | Grounded | Answered | Tok/q | p50 s | p95 s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, summary in stages:
        for split, name in SPLIT_NAMES.items():
            row = summary[split]
            lines.append(
                f"| {label} | {name} | {row['hit_rate']:.0%} | {row['correctness']:.0%} "
                f"| {row['groundedness']:.0%} | {row['answered']:.0%} "
                f"| {row['tokens_per_query']:.0f} | {row['latency_p50_s']:.1f} "
                f"| {row['latency_p95_s']:.1f} |"
            )
    return "\n".join(lines)


def inject(readme: str, block: str) -> str:
    """Replace the text between the results markers; leave the README alone if they are absent."""
    if START not in readme or END not in readme:
        return readme
    head, rest = readme.split(START, 1)
    return f"{head}{START}\n{block}\n{END}{rest.split(END, 1)[1]}"


def main() -> None:
    available = [
        (label, REPORTS / f"{name}.json")
        for name, label in STAGES
        if (REPORTS / f"{name}.json").exists()
    ]
    if len(available) < 2:
        raise SystemExit("need at least two stage reports in reports/ (run the eval targets first)")
    stages = [(label, load_summary(path)) for label, path in available]
    meta = json.loads(available[-1][1].read_text(encoding="utf-8"))["meta"]
    top_k = int(meta["top_k"])
    headline = headline_table(stages[0][1], stages[-1][1], top_k)
    caption = (
        f"Before = {available[0][0]}, after = {available[-1][0]}. "
        f"{meta['questions']} questions, LLM and judge `{meta['llm']}`, "
        f"embeddings `{meta['embeddings']}`, run locally."
    )
    (REPORTS / "before_after.md").write_text(
        f"# Before / after\n\n{caption}\n\n{headline}\n\n## Every stage\n\n"
        f"{stage_table(stages, top_k)}\n\n"
        "Grounded = supported answers / answers that made a claim. Answered = share of "
        "questions where the system made a claim instead of abstaining.\n",
        encoding="utf-8",
    )
    sentence = headline_sentence(stages[0][1], stages[-1][1])
    README.write_text(
        inject(README.read_text(encoding="utf-8"), f"{sentence}\n\n{headline}\n\n{caption}"),
        encoding="utf-8",
    )
    print(headline)


if __name__ == "__main__":
    main()
