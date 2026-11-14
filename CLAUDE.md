# Brief for Claude: build `rag-doctor-ar`

## Goal
A small public demo repo that proves one skill: diagnosing and fixing a broken bilingual (Arabic + English) RAG pipeline, with evals that show before/after numbers. It is a portfolio piece for freelance "fixed-price LLM fix" work. Keep it small, clean, and runnable on a MacBook with local models (zero API cost by default; OpenAI/Claude optional via env vars).

## Hard rules
- Use only public data. No employer code, data, prompts, or names.
- Python 3.11+, `uv` or `pip`, `make` targets. MIT license.
- Every number in the README must come from an actual `make eval` run.

## Structure
```
data/            # public docs (MSA, Gulf dialect, English), 200-500 docs + fetch script
evals/questions.jsonl  # 60-100 Qs: {id, lang: msa|gulf|en, question, gold_answer, gold_doc_ids}
rag/baseline.py  # deliberately broken pipeline
rag/fixed.py     # fixed pipeline
rag/common.py    # embeddings, vector store (local: e.g. Chroma/FAISS), LLM client (Ollama default)
evals/run.py     # scoring + report
reports/         # generated before/after markdown + json
Makefile         # make data | make eval-baseline | make eval-fixed | make report
```

## The two planted bugs (baseline)
1. **Over-aggressive post-retrieval filter**: a score/metadata filter that silently drops ~half of correct retrieved chunks.
2. **English-only chunking**: splitter that ignores Arabic sentence boundaries / normalization (alef/ya/ta-marbuta, diacritics, tatweel), hurting recall for Arabic and especially Gulf-dialect queries.

Fix each in its own commit and re-run evals after each, so history tells the story.

## Metrics (per language split + overall)
- Retrieval hit rate @k (gold doc in top-k)
- Answer correctness (LLM judge vs gold answer, with judge prompt in repo)
- Groundedness (answer supported by retrieved chunks)
- Cost (tokens) and p50/p95 latency per query

## Regression guard
Add a short section + example showing how https://github.com/AMOHYZ/langgraph-regression-guard would catch a future regression (CI-style check that fails if hit rate or correctness drops beyond a threshold).

## README (it's the sales page)
1. One-line pitch + before/after table at the top.
2. "What was broken and how I found it" in ~150 words.
3. How to run in 3 commands.
4. Closing line: "Have a RAG that gives wrong answers? I do fixed-price audits." + contact placeholder.

## Done when
- `make data && make eval-baseline && make eval-fixed && make report` works from a clean clone.
- README table is filled from real runs.
- Commit history shows: baseline → fix 1 → fix 2 → guard.
