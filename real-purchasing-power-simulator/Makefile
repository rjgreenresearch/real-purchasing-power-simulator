.PHONY: help install install-dev test test-cov data metrics clean lint typecheck all

# All Python tools are invoked as `python -m <tool>` rather than as bare
# executables. This works on Windows even when C:\Python3xx\Scripts\ is
# not on PATH, and on macOS / Linux regardless of which Python install
# (system, pyenv, conda) provides the interpreter. The `PYTHON` variable
# can be overridden, e.g. `make test PYTHON=python3.11`.
PYTHON ?= python

help:
	@echo "Real Purchasing Power Simulator - Makefile targets"
	@echo ""
	@echo "  make install      install runtime dependencies and the rpps package"
	@echo "  make install-dev  install runtime + development dependencies"
	@echo "  make test         run the test suite (offline; uses fixture data)"
	@echo "  make test-cov     run the test suite with coverage report"
	@echo "  make data         download FRED + NBER series and build spliced dataset"
	@echo "                    requires FRED_API_KEY environment variable"
	@echo "  make metrics      compute RPPH, WICR, PRWDI from the spliced dataset"
	@echo "                    requires 'make data' to have been run first"
	@echo "  make lint         run ruff linter"
	@echo "  make typecheck    run mypy static type checker"
	@echo "  make clean        remove caches and processed outputs"
	@echo "  make all          install-dev + lint + typecheck + test"
	@echo ""
	@echo "Override the Python interpreter with PYTHON=...:"
	@echo "  make test PYTHON=python3.11"

install:
	$(PYTHON) -m pip install -e . --break-system-packages

install-dev:
	$(PYTHON) -m pip install -e ".[dev]" --break-system-packages

test:
	$(PYTHON) -m pytest tests/

test-cov:
	$(PYTHON) -m pytest tests/ --cov=rpps --cov-report=term-missing --cov-report=html

data:
	$(PYTHON) -m rpps.fred_loader --download-all
	$(PYTHON) -m rpps.nber_splice --build-spliced-dataset

metrics:
	$(PYTHON) -m rpps.metrics.compute_all --output data/processed

lint:
	$(PYTHON) -m ruff check rpps/ tests/

typecheck:
	$(PYTHON) -m mypy rpps/

clean:
	rm -rf data/raw/fred/*.csv data/raw/fred/*.json
	rm -rf data/processed/*.csv data/processed/*.json
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: install-dev lint typecheck test
