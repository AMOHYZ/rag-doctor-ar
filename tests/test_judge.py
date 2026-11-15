from __future__ import annotations

from evals.judge import JUDGE_ERROR, judge_correctness, judge_groundedness, parse_verdict
from rag.common import NO_CONTEXT_ANSWER, Chunk, Hit
from tests.fakes import FakeLLM


def test_parse_verdict_accepts_json_and_rejects_garbage() -> None:
    allowed = ("correct", "incorrect")
    assert parse_verdict('{"verdict": "correct"}', allowed) == "correct"
    assert parse_verdict('Sure!\n{"verdict": "incorrect"} done', allowed) == "incorrect"
    assert parse_verdict('{"verdict": "maybe"}', allowed) == JUDGE_ERROR
    assert parse_verdict("no json here", allowed) == JUDGE_ERROR


def test_correctness_prompt_carries_question_gold_and_answer() -> None:
    llm = FakeLLM('{"verdict": "correct"}')
    assert judge_correctness(llm, "وين تقع جدة؟", "على البحر الأحمر", "البحر الأحمر") == "correct"
    assert all(part in llm.prompts[0] for part in ("وين تقع جدة؟", "على البحر الأحمر"))


def test_groundedness_without_passages_is_no_answer_and_skips_the_llm() -> None:
    llm = FakeLLM('{"verdict": "supported"}')
    assert judge_groundedness(llm, "q", "a", []) == "no_answer"
    assert llm.prompts == []
    hit = Hit(Chunk("d", "en", "Jeddah is on the Red Sea."), 0.8)
    assert judge_groundedness(llm, "q", "a", [hit]) == "supported"
    assert "Jeddah is on the Red Sea." in llm.prompts[0]


def test_canned_abstention_is_incorrect_without_asking_the_judge() -> None:
    llm = FakeLLM('{"verdict": "correct"}')
    assert judge_correctness(llm, "q", "gold", NO_CONTEXT_ANSWER) == "incorrect"
    assert llm.prompts == []
