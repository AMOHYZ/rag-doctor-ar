# Eval report: Fix 1: filter

60 questions, 1164 chunks, LLM `ollama/qwen3:8b`, embeddings `bge-m3`, generated 2026-11-14T17:40:18+00:00.

| Split | n | Hit@5 | Correct | Grounded | Answered | Tokens/q | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|
| msa | 20 | 100% | 15% | 75% | 60% | 2221 | 47.6 | 57.3 |
| gulf | 20 | 100% | 5% | 50% | 60% | 1835 | 31.0 | 52.8 |
| en | 20 | 100% | 95% | 100% | 100% | 908 | 16.1 | 21.8 |
| overall | 60 | 100% | 38% | 80% | 73% | 1654 | 31.0 | 54.0 |

Grounded = supported answers / answers that made a claim. Answered = share of questions where the system made a claim instead of abstaining. Tokens and latency cover retrieval and generation only (not the judge).
