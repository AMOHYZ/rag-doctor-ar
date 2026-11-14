# Optional overrides live in .env (git-ignored); see .env.example.
-include .env
export

LLM_MODEL_PULL ?= qwen3:8b
EMBED_MODEL ?= bge-m3
PY := uv run python

.PHONY: setup smoke data test lint

setup:  ## install deps, pull local models, run a smoke query
	uv sync
	ollama pull $(EMBED_MODEL)
	ollama pull $(LLM_MODEL_PULL)
	$(MAKE) smoke

smoke:  ## embed three tiny docs and answer one question end to end
	$(PY) -m rag.smoke

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check . && uv run mypy rag evals tests
