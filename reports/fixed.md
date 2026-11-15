# Eval report: Fixed

60 questions, 2423 chunks, LLM `ollama/qwen3:8b`, embeddings `bge-m3`, generated 2026-11-14T19:25:46+00:00.

| Split | n | Hit@5 | Correct | Grounded | Answered | Tokens/q | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|
| msa | 20 | 100% | 100% | 100% | 100% | 1258 | 25.8 | 29.0 |
| gulf | 20 | 100% | 80% | 100% | 95% | 1262 | 25.2 | 28.9 |
| en | 20 | 100% | 95% | 100% | 100% | 902 | 16.8 | 19.4 |
| overall | 60 | 100% | 92% | 100% | 98% | 1141 | 23.6 | 28.9 |

Grounded = supported answers / answers that made a claim. Answered = share of questions where the system made a claim instead of abstaining. Tokens and latency cover retrieval and generation only (not the judge).
