# Eval report: Baseline (broken)

60 questions, 1164 chunks, LLM `ollama/qwen3:8b`, embeddings `bge-m3`, generated 2026-11-14T15:29:45+00:00.

| Split | n | Hit@5 | Correct | Grounded | Answered | Tokens/q | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|
| msa | 20 | 65% | 5% | 71% | 35% | 698 | 11.7 | 47.0 |
| gulf | 20 | 50% | 10% | 60% | 25% | 322 | 8.3 | 16.0 |
| en | 20 | 100% | 95% | 100% | 100% | 600 | 11.8 | 21.4 |
| overall | 60 | 72% | 37% | 88% | 53% | 540 | 10.5 | 28.0 |

Grounded = supported answers / answers that made a claim. Answered = share of questions where the system made a claim instead of abstaining. Tokens and latency cover retrieval and generation only (not the judge).
