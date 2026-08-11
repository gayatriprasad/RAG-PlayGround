.PHONY: setup dev test lint eval services-up services-down clean ollama-setup ollama-serve

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
	@$(MAKE) ollama-setup
	@echo ""
	@echo "Setup complete. Run 'make dev'."

ollama-setup:
	@if ! command -v zstd >/dev/null 2>&1; then \
		echo "Installing zstd (required by the Ollama installer)..."; \
		sudo apt-get update -qq && sudo apt-get install -y -qq zstd; \
	fi
	@if ! command -v ollama >/dev/null 2>&1; then \
		echo "Installing Ollama..."; \
		curl -fsSL https://ollama.com/install.sh | sh; \
	else \
		echo "Ollama already installed."; \
		if ! test -x /usr/local/lib/ollama/llama-server \
			&& ! test -x /usr/local/bin/build/lib/ollama/llama-server \
			&& ! test -x /usr/local/bin/dist/linux-arm64/lib/ollama/llama-server \
			&& ! test -x /usr/local/bin/dist/linux_arm64/lib/ollama/llama-server; then \
			echo "Detected incomplete Ollama install (missing llama-server). Reinstalling Ollama..."; \
			curl -fsSL https://ollama.com/install.sh | sh; \
		fi; \
	fi
	@$(MAKE) ollama-serve
	@# llama3.2:1b (~1.3 GiB) is the default — runs in any container with ≥4 GiB RAM.
	@# llama3 (8B, ~4.7 GiB) only pulled when ≥12 GiB RAM available to avoid OOM kills.
	@if ! ollama list 2>/dev/null | grep -q "^llama3.2:1b"; then \
		echo "Pulling llama3.2:1b (default small model, ~1.3 GiB)..."; \
		ollama pull llama3.2:1b; \
	else \
		echo "llama3.2:1b already pulled."; \
	fi
	@TOTAL_MEM=$$(awk '/MemTotal/{print int($$2/1024/1024)}' /proc/meminfo); \
	if [ "$$TOTAL_MEM" -ge 12 ]; then \
		if ! ollama list 2>/dev/null | grep -q "^llama3:latest"; then \
			echo "System has $${TOTAL_MEM}GB RAM — also pulling llama3 (8B)..."; \
			ollama pull llama3; \
		else \
			echo "llama3 already pulled."; \
		fi; \
	else \
		echo "System has $${TOTAL_MEM}GB RAM (<12GB) — skipping llama3 8B pull to avoid OOM."; \
	fi

ollama-serve:
	@if ! command -v ollama >/dev/null 2>&1; then \
		echo "Warning: ollama not installed — run 'make setup' first. Ollama-backed presets will fail."; \
	elif ! curl -sS http://localhost:11434/api/tags >/dev/null 2>&1; then \
		echo "Starting Ollama daemon..."; \
		nohup ollama serve > /tmp/ollama.log 2>&1 & \
		for i in $$(seq 1 15); do \
			curl -sS http://localhost:11434/api/tags >/dev/null 2>&1 && break; \
			sleep 1; \
		done; \
	else \
		echo "Ollama daemon already running."; \
	fi

dev:
	@$(MAKE) ollama-serve
	@echo "Starting API (:8001) + frontend (:3000) — Ctrl-C stops both"
	$(VENV)/bin/uvicorn api.main:app --port 8001 --reload &
	cd app && npm run dev

test:
	cd rag-lab && ../$(PY) -m pytest tests/ --cov=raglab --cov-report=term-missing --cov-fail-under=80

lint:
	cd rag-lab && ../$(VENV)/bin/ruff check src/ && ../$(VENV)/bin/mypy src/raglab --ignore-missing-imports

eval:
	@$(MAKE) ollama-serve
	cd rag-lab && ../$(PY) -m raglab.run_experiment --config experiments/02_retrieval_comparison/config.yaml

services-up:
	docker compose -f docker/compose.yml up -d

services-down:
	docker compose -f docker/compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf rag-lab/out/chroma rag-lab/.pytest_cache

mutation-test:
	cd rag-lab && ../.venv/bin/python -m mutmut run
