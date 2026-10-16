PYTHON ?= python

.PHONY: install test lint typecheck format schemas example

install:
	$(PYTHON) -m pip install -e ".[dev,llm]"

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
