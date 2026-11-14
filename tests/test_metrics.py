from __future__ import annotations

from evals.metrics import Record, percentile, summarize, to_markdown


def _record(lang: str, hit: bool, correctness: str, groundedness: str, latency: float) -> Record:
    return Record(
        id="q",
        lang=lang,
        question="?",
        gold_answer="a",
        gold_doc_ids=("d",),
        answer="a",
        retrieved_doc_ids=("d",) if hit else (),
        scores=(0.5,) if hit else (),
        hit=hit,
        correctness=correctness,
        groundedness=groundedness,
        prompt_tokens=90,
        completion_tokens=10,
        latency_s=latency,
    )


RECORDS = [
    _record("msa", True, "correct", "supported", 1.0),
    _record("msa", False, "incorrect", "no_answer", 2.0),
    _record("gulf", True, "incorrect", "unsupported", 3.0),
    _record("en", True, "correct", "supported", 4.0),
]


def test_percentile_nearest_rank() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 95) == 4.0
    assert percentile([], 50) == 0.0


def test_summarize_per_split_and_overall() -> None:
    rows = {row.split: row for row in summarize(RECORDS)}
    assert list(rows) == ["msa", "gulf", "en", "overall"]
    assert rows["msa"].hit_rate == 0.5 and rows["msa"].correctness == 0.5
    assert rows["msa"].answered == 0.5 and rows["msa"].groundedness == 1.0
    assert rows["gulf"].groundedness == 0.0
    overall = rows["overall"]
    assert overall.n == 4 and overall.hit_rate == 0.75 and overall.correctness == 0.5
    assert overall.groundedness == 2 / 3 and overall.tokens_per_query == 100
    assert overall.latency_p50_s == 2.0 and overall.latency_p95_s == 4.0


def test_markdown_table_has_one_row_per_split() -> None:
    table = to_markdown(summarize(RECORDS), top_k=5)
    assert table.splitlines()[0].startswith("| Split | n | Hit@5 |")
    assert len(table.splitlines()) == 6
    assert "| overall | 4 | 75% | 50% | 67% | 75% | 100 | 2.0 | 4.0 |" in table
