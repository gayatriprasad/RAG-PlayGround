.PHONY: setup dev test lint eval services-up services-down clean

# Single venv at repo root (NOT rag-lab/.venv) — matches this repo's actual convention.
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e "rag-lab/.[agents,observability,dev]"
	cd app && npm ci
	cp -n rag-lab/.env.example rag-lab/.env || true
	$(VENV)/bin/pre-commit install
	@echo ""
	@echo "Pull a local model if you don't have one yet: ollama pull llama3"
	@echo "Setup complete. Run 'make dev'."

dev:
	@echo "Starting API (:8001) + frontend (:3000) — Ctrl-C stops both"
	$(VENV)/bin/uvicorn api.main:app --port 8001 --reload &
	cd app && npm run dev

test:
	cd rag-lab && ../$(PY) -m pytest tests/ --cov=raglab --cov-report=term-missing --cov-fail-under=80

lint:
	cd rag-lab && ../$(VENV)/bin/ruff check src/ && ../$(VENV)/bin/mypy src/raglab --ignore-missing-imports

eval:
	cd rag-lab && ../$(PY) -m raglab.run_experiment --config experiments/02_retrieval_comparison/config.yaml

services-up:
	docker compose -f docker/compose.yml up -d

services-down:
	docker compose -f docker/compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf rag-lab/out/chroma rag-lab/.pytest_cache
