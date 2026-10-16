PYTHON ?= python

.PHONY: install test lint typecheck format schemas example construct-example run-example run-all serve stores-up stores-down

install:
	$(PYTHON) -m pip install -e ".[dev,llm,api]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts examples
	$(PYTHON) -m mypy

format:
	$(PYTHON) -m ruff check --fix src tests scripts examples
	$(PYTHON) -m ruff format src tests scripts examples

schemas:
	$(PYTHON) scripts/export_schemas.py

example:
	$(PYTHON) examples/run_example.py

construct-example:
	$(PYTHON) examples/run_construction_example.py

# Stage 1 -> 2 -> 3 offline on the generated sample PDF; fails unless every answer is verified.
run-example:
	$(PYTHON) -m retrieval run-all --offline --out examples/output/run_all

# Same chain with the providers in config.yaml / construction.yaml / retrieval.yaml.
run-all:
	$(PYTHON) -m retrieval run-all --out output/run_all

serve:
	$(PYTHON) -m retrieval serve

stores-up:
	docker compose up -d

stores-down:
	docker compose down
