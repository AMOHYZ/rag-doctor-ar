# rag-doctor-ar

**A bilingual (Arabic + English) RAG pipeline that gave wrong answers in Arabic, diagnosed and fixed, with evals that show the before and after.** Everything runs on a laptop with local models.

<!-- results:start -->
**Answer correctness: 37% → 92% overall** (MSA 5% → 100%, Gulf dialect 10% → 80%, English 95% → 95%).

| Questions | Hit@5 | Answer correct | Grounded | Tokens/query | p95 latency |
|---|---|---|---|---|---|
| MSA (20) | 65% → 100% | 5% → 100% | 71% → 100% | 698 → 1258 | 47.0s → 29.0s |
| Gulf dialect (20) | 50% → 100% | 10% → 80% | 60% → 100% | 322 → 1262 | 16.0s → 28.9s |
| English (20) | 100% → 100% | 95% → 95% | 100% → 100% | 600 → 902 | 21.4s → 19.4s |
| **Overall** (60) | 72% → 100% | 37% → 92% | 88% → 100% | 540 → 1141 | 28.0s → 28.9s |

Before = Baseline (broken), after = Fix 2: + Arabic chunking. 60 questions, LLM and judge `ollama/qwen3:8b`, embeddings `bge-m3`, run locally.
<!-- results:end -->

The block above is written by `make report` from the eval runs in [`reports/`](reports/). Numbers for every stage are in [`reports/before_after.md`](reports/before_after.md).

## What was broken and how I found it

The baseline has two bugs, planted on purpose. Both are the kind that appear when a pipeline is built and tested in English.

The per-language split gave the first one away: English was 95% correct, MSA 5%, Gulf dialect 10%. The *Answered* column showed why. The system abstained on most Arabic questions because it had no context: a post-retrieval filter dropped every chunk below a fixed similarity score, and Arabic queries score lower than English ones. **Fix 1** ranks and keeps the top 5 instead.

Hit rate went to 100%, yet correctness stayed at 38% and Arabic prompts more than tripled in size. That pointed at the chunks. The splitter ended sentences only at ASCII punctuation followed by a capital letter, so each Arabic article became one giant chunk, clipped before the answer when the prompt was built. **Fix 2** splits on Arabic punctuation and normalises alef, ya, ta-marbuta, diacritics and tatweel. Correctness: 92%.

## Run it in 3 commands

```bash
make setup                             # install deps, pull the two Ollama models, smoke test
make data                              # download the pinned public corpus
make eval-baseline eval-fixed report   # run both pipelines and write the before/after table
```

You need [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com). The defaults cost nothing: `qwen3:8b` answers and judges, `bge-m3` embeds. `make data` takes about half an hour because Wikipedia throttles anonymous clients, and each eval run takes up to two hours on a laptop CPU, mostly in the judge.

To use a hosted model instead, copy `.env.example` to `.env` and set `LLM_PROVIDER` to `openai` or `anthropic` with the matching key. The committed reports were produced with the local defaults only.

## What is in the repo

| Path | What it is |
|---|---|
| [`rag/baseline.py`](rag/baseline.py) | The broken pipeline. Both bugs are marked in the code. |
| [`rag/fixed.py`](rag/fixed.py) | The fixed pipeline, same shape. |
| [`rag/graph.py`](rag/graph.py) | The pipeline as a LangGraph graph: `retrieve` → `post_filter` → `generate`. |
| [`rag/common.py`](rag/common.py) | Embeddings, FAISS vector store, LLM clients, prompt. |
| [`data/`](data/) | 313 pinned Wikipedia articles (182 Arabic, 131 English) and the fetch script. |
| [`evals/questions.jsonl`](evals/questions.jsonl) | 60 questions: 20 MSA, 20 Gulf dialect, 20 English, each with a gold answer and gold document. |
| [`evals/run.py`](evals/run.py) | Scoring. Judge prompts are in [`evals/prompts/`](evals/prompts/). |
| [`reports/`](reports/) | Generated reports, markdown and JSON, with every answer and verdict. |

The git history tells the same story, one commit per step: baseline, fix 1, fix 2, guard.

Metrics, per language split and overall:

- **Hit@5**: the gold document is among the chunks handed to the model.
- **Answer correct**: an LLM judge compares the answer with the gold answer.
- **Grounded**: an LLM judge checks that the answer is supported by the retrieved chunks. Abstentions are not counted.
- **Tokens/query** and **latency** (p50 and p95) for retrieval and generation.

## Regression guard

Two checks keep the fix fixed. Both run in CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

**1. Did a metric drop?** `make guard` compares the latest eval report with the accepted reference and fails if hit rate or answer correctness is more than 5 points lower on any split. Pointed at the broken baseline, it fails like this:

```text
$ make guard CANDIDATE=reports/baseline.json
guard FAILED: reports/baseline.json regressed vs reports/guard_reference.json

| split | metric | reference | candidate | drop |
|---|---|---|---|---|
| msa | hit_rate | 100% | 65% | -35% |
| msa | correctness | 100% | 5% | -95% |
| gulf | hit_rate | 100% | 50% | -50% |
| gulf | correctness | 80% | 10% | -70% |
| overall | hit_rate | 100% | 72% | -28% |
| overall | correctness | 92% | 37% | -55% |
```

**2. Which step broke?** Because the pipeline is a LangGraph graph, [langgraph-regression-guard](https://github.com/AMOHYZ/langgraph-regression-guard) (`lgcov`) can replay recorded runs and name the node whose output changed first. Put the score cutoff back into `filter_hits` in `rag/fixed.py` and it reports:

```text
$ make guard-replay
### Cassette replay

| node | drift | runs |
|---|---|---|
| `generate` | propagated | 3/3 |
| `post_filter` | root | 3/3 |

Unchanged: `retrieve`
```

The filter is the root; the worse answers are a consequence. CI runs the same replay with `make guard-ci`. Unchanged nodes are served from the recordings in [`cassettes/`](cassettes/), so that job needs no model. After an intended change, re-record with `make guard-record` and accept new eval numbers with `make guard-accept`.

## Limits

- The eval set is small. With 20 questions per split, one question is 5 points.
- Every question names its subject, so Hit@5 reaches 100% as soon as the filter is fixed. The chunking fix shows in answer correctness, not in hit rate.
- The judge is the same local 8B model, run in thinking mode. Its verdicts are in the JSON reports so you can audit them. Set `JUDGE_PROVIDER` and `JUDGE_MODEL` for a stronger judge.
- The documents are MSA and English. Gulf dialect appears in the questions only.
- Latency was measured with Ollama on a laptop CPU. Read it as relative, not absolute.

Corpus text is from Wikipedia, CC BY-SA 4.0; see [`data/README.md`](data/README.md). Code is MIT.

---

**Have a RAG that gives wrong answers? I do fixed-price audits.** Contact: `<your contact here>`
