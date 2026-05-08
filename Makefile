.PHONY: help install install-dev test test-cov data metrics clean lint typecheck all

help:
	@echo "Real Purchasing Power Simulator — Makefile targets"
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

install:
	pip install -e . --break-system-packages

install-dev:
	pip install -e ".[dev]" --break-system-packages

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=rpps --cov-report=term-missing --cov-report=html

data:
	@if [ -z "$$FRED_API_KEY" ]; then \
		echo "ERROR: FRED_API_KEY is not set."; \
		echo "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"; \
		exit 1; \
	fi
	python -m rpps.fred_loader --download-all
	python -m rpps.nber_splice --build-spliced-dataset

metrics:
	python -m rpps.metrics.compute_all --output data/processed

lint:
	ruff check rpps/ tests/

typecheck:
	mypy rpps/

clean:
	rm -rf data/raw/fred/*.csv data/raw/fred/*.json
	rm -rf data/processed/*.csv data/processed/*.json
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: install-dev lint typecheck test
