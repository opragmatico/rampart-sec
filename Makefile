PYTHON ?= python3.13
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: help venv install lint format typecheck test test-live cov serve index corpus eval docker-up docker-down clean

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv:            ## create the virtual environment
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: venv    ## install the package with the dev tools
	$(BIN)/pip install -e ".[dev]"

lint:            ## ruff check + format check
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:          ## apply ruff fixes and formatting
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

typecheck:       ## mypy (advisory)
	$(BIN)/mypy

test:            ## unit + recorded tests (no network, no tokens)
	$(BIN)/pytest

cov:             ## tests with a coverage report
	$(BIN)/pytest --cov --cov-report=term-missing

test-live:       ## the six validation scenarios against the real APIs (needs keys, ~3 min)
	$(BIN)/pytest -m live -v

serve:           ## run the service on RAMPART_HOST:RAMPART_PORT
	$(BIN)/rampart serve

index:           ## embed the corpus into Qdrant (incremental)
	$(BIN)/rampart index

corpus:          ## regenerate the corpus PDFs from the YAML specs (verifies every CVE against NVD)
	$(BIN)/rampart build-corpus

eval:            ## signature-guard precision/recall and retrieval hit rate
	$(BIN)/rampart eval guardrail
	$(BIN)/rampart eval retrieval

docker-up:       ## Qdrant + Rampart
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
