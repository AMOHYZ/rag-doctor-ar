"""Schema checks for the committed eval set (offline: uses the manifest, not the corpus)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = [
    json.loads(line)
    for line in (ROOT / "evals" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
MANIFEST = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))


def test_schema_and_unique_ids() -> None:
    fields = {"id", "lang", "question", "gold_answer", "gold_doc_ids"}
    assert all(set(item) == fields for item in QUESTIONS)
    assert len({item["id"] for item in QUESTIONS}) == len(QUESTIONS)
    assert all(item["question"].strip() and item["gold_answer"].strip() for item in QUESTIONS)


def test_size_and_language_splits() -> None:
    counts = Counter(item["lang"] for item in QUESTIONS)
    assert 60 <= len(QUESTIONS) <= 100
    assert set(counts) == {"msa", "gulf", "en"}
    assert min(counts.values()) >= 20


def test_gold_docs_exist_and_match_the_question_language() -> None:
    doc_lang = {entry["id"]: entry["lang"] for entry in MANIFEST}
    for item in QUESTIONS:
        assert item["gold_doc_ids"], item["id"]
        expected = "en" if item["lang"] == "en" else "ar"
        assert all(doc_lang.get(doc_id) == expected for doc_id in item["gold_doc_ids"]), item["id"]
