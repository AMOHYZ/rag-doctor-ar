# Before / after

Before = Baseline (broken), after = Fix 2: + Arabic chunking. 60 questions, LLM and judge `ollama/qwen3:8b`, embeddings `bge-m3`, run locally.

| Questions | Hit@5 | Answer correct | Grounded | Tokens/query | p95 latency |
|---|---|---|---|---|---|
| MSA (20) | 65% → 100% | 5% → 100% | 71% → 100% | 698 → 1258 | 47.0s → 29.0s |
| Gulf dialect (20) | 50% → 100% | 10% → 80% | 60% → 100% | 322 → 1262 | 16.0s → 28.9s |
| English (20) | 100% → 100% | 95% → 95% | 100% → 100% | 600 → 902 | 21.4s → 19.4s |
| **Overall** (60) | 72% → 100% | 37% → 92% | 88% → 100% | 540 → 1141 | 28.0s → 28.9s |

## Every stage

| Stage | Split | Hit@5 | Correct | Grounded | Answered | Tok/q | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|
| Baseline (broken) | MSA | 65% | 5% | 71% | 35% | 698 | 11.7 | 47.0 |
| Baseline (broken) | Gulf dialect | 50% | 10% | 60% | 25% | 322 | 8.3 | 16.0 |
| Baseline (broken) | English | 100% | 95% | 100% | 100% | 600 | 11.8 | 21.4 |
| Baseline (broken) | **Overall** | 72% | 37% | 88% | 53% | 540 | 10.5 | 28.0 |
| Fix 1: filter | MSA | 100% | 15% | 75% | 60% | 2221 | 47.6 | 57.3 |
| Fix 1: filter | Gulf dialect | 100% | 5% | 50% | 60% | 1835 | 31.0 | 52.8 |
| Fix 1: filter | English | 100% | 95% | 100% | 100% | 908 | 16.1 | 21.8 |
| Fix 1: filter | **Overall** | 100% | 38% | 80% | 73% | 1654 | 31.0 | 54.0 |
| Fix 2: + Arabic chunking | MSA | 100% | 100% | 100% | 100% | 1258 | 25.8 | 29.0 |
| Fix 2: + Arabic chunking | Gulf dialect | 100% | 80% | 100% | 95% | 1262 | 25.2 | 28.9 |
| Fix 2: + Arabic chunking | English | 100% | 95% | 100% | 100% | 902 | 16.8 | 19.4 |
| Fix 2: + Arabic chunking | **Overall** | 100% | 92% | 100% | 98% | 1141 | 23.6 | 28.9 |

Grounded = supported answers / answers that made a claim. Answered = share of questions where the system made a claim instead of abstaining.
