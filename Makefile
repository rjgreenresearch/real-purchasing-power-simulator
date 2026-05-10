.PHONY: help install install-dev test test-cov data metrics phase3 report clean lint typecheck all

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
	@echo "  make phase3       run break detection, regime regression, counterfactual"
	@echo "                    requires 'make metrics' to have been run first"
	@echo "  make phase3-quick faster phase3 with n_bootstrap=200 (laptop iteration)"
	@echo "  make phase3-permissive  phase3 with break-penalty-scale=0.10"
	@echo "                    (typically detects 1972/1982 canonical regimes on k=4 panel)"
	@echo "  make report       generate a self-contained HTML analysis report"
	@echo "                    if phase3 outputs exist, the report includes them"
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

phase3:
	$(PYTHON) run_phase3.py

phase3-quick:
	$(PYTHON) run_phase3.py --quick

phase3-permissive:
	$(PYTHON) run_phase3.py --break-penalty-scale 0.10

report:
	$(PYTHON) -m rpps.report --processed-dir data/processed --output report.html
	@echo ""
	@echo "Report written to: report.html"
	@echo "Open in any browser; the report is a self-contained single file."

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
