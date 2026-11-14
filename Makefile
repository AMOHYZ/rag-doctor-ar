# Optional overrides live in .env (git-ignored); see .env.example.
-include .env
export

LLM_MODEL_PULL ?= qwen3:8b
EMBED_MODEL ?= bge-m3
PY := uv run python
CANDIDATE ?= reports/fixed.json

.PHONY: setup smoke data test lint eval-baseline eval-fixed report guard guard-accept guard-record guard-replay guard-ci

setup:  ## install deps, pull local models, run a smoke query
	uv sync
	ollama pull $(EMBED_MODEL)
	ollama pull $(LLM_MODEL_PULL)
	$(MAKE) smoke

smoke:  ## embed three tiny docs and answer one question end to end
	$(PY) -m rag.smoke

data:  ## download the pinned public corpus into data/corpus.jsonl
	uv sync
	$(PY) -m data.fetch

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check . && uv run mypy rag evals tests data

eval-baseline:  ## run the broken pipeline over the question set
	$(PY) -m evals.run --pipeline baseline --out reports/baseline --label "Baseline (broken)"

eval-fixed:  ## run the fixed pipeline over the question set
	$(PY) -m evals.run --pipeline fixed --out reports/fixed --label "Fixed"

report:  ## before/after table -> reports/before_after.md and the README results block
	$(PY) -m evals.report

guard:  ## fail if hit rate or correctness dropped vs the accepted reference
	$(PY) -m evals.guard --reference reports/guard_reference.json --candidate $(CANDIDATE)

guard-accept:  ## accept the current fixed report as the new reference
	cp reports/fixed.json reports/guard_reference.json

guard-record:  ## record lgcov cassettes for the current graph (needs Ollama)
	PYTHONPATH=. uv run --with lgcov==0.3.0 lgcov record --inputs evals/lgcov_inputs.json

guard-replay:  ## replay the cassettes; exits 1 and names the node if a node's output drifted
	PYTHONPATH=. uv run --with lgcov==0.3.0 lgcov replay --fail-on root-drift

guard-ci:  ## what CI runs: the same replay, offline; changed nodes that need a model are not run
	mkdir -p .cache
	PYTHONPATH=. uv run --with lgcov==0.3.0 lgcov ci --report .cache/lgcov_report.md \
		--on-divergence fail --fail-on root-drift
