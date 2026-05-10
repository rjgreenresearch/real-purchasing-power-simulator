"""Tests for rpps.metrics.compute_all — Phase 2 batch orchestrator.

These tests verify the orchestration contract (per-metric isolation, run
summary structure, file outputs) without requiring a live FRED API key. The
upstream functions are monkeypatched to return deterministic synthetic data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rpps.metrics import compute_all

# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _synthetic_wages() -> pd.Series:
    idx = pd.date_range("1990-01-31", "2024-12-31", freq="ME")
    values = np.linspace(10.0, 30.0, len(idx))
    return pd.Series(values, index=idx, name="AHETPI_spliced")


def _synthetic_cpi() -> pd.Series:
    idx = pd.date_range("1990-01-31", "2024-12-31", freq="ME")
    # 3% annual compound.
    monthly = 1.03 ** (1.0 / 12.0)
    values = 100.0 * np.power(monthly, np.arange(len(idx)))
    return pd.Series(values, index=idx, name="CPIAUCNS")


def _synthetic_productivity() -> pd.Series:
    idx = pd.date_range("1947-03-31", "2024-12-31", freq="QE")
    quarterly = 1.025 ** (1.0 / 4.0)
    values = 100.0 * np.power(quarterly, np.arange(len(idx)))
    return pd.Series(values, index=idx, name="OPHNFB")


def _synthetic_real_compensation() -> pd.Series:
    idx = pd.date_range("1947-03-31", "2024-12-31", freq="QE")
    quarterly = 1.015 ** (1.0 / 4.0)
    values = 100.0 * np.power(quarterly, np.arange(len(idx)))
    return pd.Series(values, index=idx, name="COMPRNFB")


def _synthetic_basket_panel() -> pd.DataFrame:
    idx = pd.date_range("1990-01-31", "2024-12-31", freq="ME")
    n = len(idx)
    return pd.DataFrame(
        {
            "gasoline": np.linspace(15.0, 40.0, n),
            "beef": np.linspace(60.0, 230.0, n),
            "electricity": np.linspace(80.0, 150.0, n),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Patched run
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_run(monkeypatch):
    """Monkeypatch FRED/NBER loaders to return synthetic data."""
    def fake_load_series(series_id: str, *args, **kwargs):
        if series_id == "CPIAUCNS":
            return _synthetic_cpi()
        if series_id == "OPHNFB":
            return _synthetic_productivity()
        if series_id == "COMPRNFB":
            return _synthetic_real_compensation()
        raise KeyError(f"unexpected series_id in test: {series_id}")

    def fake_load_spliced_wages(*args, **kwargs):
        return _synthetic_wages()

    def fake_basket_cost_panel(*args, **kwargs):
        return _synthetic_basket_panel()

    monkeypatch.setattr("rpps.metrics.compute_all.load_series", fake_load_series)
    monkeypatch.setattr(
        "rpps.metrics.compute_all.load_spliced_wages", fake_load_spliced_wages,
    )
    monkeypatch.setattr(
        "rpps.metrics.compute_all.basket_cost_panel", fake_basket_cost_panel,
    )


# ---------------------------------------------------------------------------
# Run-summary contract
# ---------------------------------------------------------------------------

class TestRunSummary:
    def test_run_returns_dict(self, tmp_path: Path, patched_run):
        summary = compute_all.run(output_dir=tmp_path)
        assert isinstance(summary, dict)

    def test_summary_has_required_keys(self, tmp_path: Path, patched_run):
        summary = compute_all.run(output_dir=tmp_path)
        for key in (
            "rpps_version", "started_at_utc", "finished_at_utc",
            "output_dir", "frequency", "base_year", "results",
            "n_ok", "n_error", "run_summary_path",
        ):
            assert key in summary, f"missing key: {key}"

    def test_results_contain_three_metrics(self, tmp_path: Path, patched_run):
        summary = compute_all.run(output_dir=tmp_path)
        names = {r["metric"] for r in summary["results"]}
        assert names == {"RPPH", "WICR", "PRWDI"}

    def test_all_three_succeed_under_synthetic_data(
        self, tmp_path: Path, patched_run,
    ):
        summary = compute_all.run(output_dir=tmp_path)
        assert summary["n_ok"] == 3
        assert summary["n_error"] == 0

    def test_summary_json_written(self, tmp_path: Path, patched_run):
        summary = compute_all.run(output_dir=tmp_path)
        path = Path(summary["run_summary_path"])
        assert path.exists()
        with open(path) as fh:
            on_disk = json.load(fh)
        assert on_disk["n_ok"] == 3


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

class TestOutputs:
    def test_rpph_outputs_written(self, tmp_path: Path, patched_run):
        compute_all.run(output_dir=tmp_path)
        assert (tmp_path / "rpph_composite.csv").exists()
        assert (tmp_path / "rpph_by_item.csv").exists()
        assert (tmp_path / "rpph_audit.json").exists()

    def test_wicr_outputs_written(self, tmp_path: Path, patched_run):
        compute_all.run(output_dir=tmp_path)
        assert (tmp_path / "wicr_panel.csv").exists()
        assert (tmp_path / "wicr_audit.json").exists()

    def test_prwdi_outputs_written(self, tmp_path: Path, patched_run):
        compute_all.run(output_dir=tmp_path)
        assert (tmp_path / "prwdi_panel.csv").exists()
        assert (tmp_path / "prwdi_audit.json").exists()


# ---------------------------------------------------------------------------
# Per-metric isolation: one failure does not break others
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_wicr_failure_does_not_block_rpph_or_prwdi(
        self, tmp_path: Path, monkeypatch,
    ):
        # Patch CPI to raise, but leave wages, productivity, and compensation valid.
        def fake_load_series(series_id: str, *args, **kwargs):
            if series_id == "CPIAUCNS":
                raise RuntimeError("simulated FRED outage")
            if series_id == "OPHNFB":
                return _synthetic_productivity()
            if series_id == "COMPRNFB":
                return _synthetic_real_compensation()
            raise KeyError(series_id)

        monkeypatch.setattr(
            "rpps.metrics.compute_all.load_series", fake_load_series,
        )
        monkeypatch.setattr(
            "rpps.metrics.compute_all.load_spliced_wages",
            lambda *a, **kw: _synthetic_wages(),
        )
        monkeypatch.setattr(
            "rpps.metrics.compute_all.basket_cost_panel",
            lambda *a, **kw: _synthetic_basket_panel(),
        )

        summary = compute_all.run(output_dir=tmp_path)
        assert summary["n_error"] == 1
        assert summary["n_ok"] == 2
        wicr_result = next(r for r in summary["results"] if r["metric"] == "WICR")
        assert wicr_result["status"] == "error"
        assert "simulated FRED outage" in wicr_result["error"]

        # Other metrics still produced files.
        assert (tmp_path / "rpph_composite.csv").exists()
        assert (tmp_path / "prwdi_panel.csv").exists()


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class TestParameters:
    def test_alternate_base_year(self, tmp_path: Path, patched_run):
        summary = compute_all.run(output_dir=tmp_path, base_year=1960)
        prwdi_result = next(r for r in summary["results"] if r["metric"] == "PRWDI")
        assert prwdi_result["status"] == "ok"
        assert prwdi_result["base_year"] == 1960

    def test_alternate_frequency(self, tmp_path: Path, patched_run):
        summary = compute_all.run(output_dir=tmp_path, frequency="Q")
        assert summary["frequency"] == "Q"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_main_returns_zero_on_success(
        self, tmp_path: Path, patched_run, capsys,
    ):
        rc = compute_all.main(
            ["--output", str(tmp_path), "--quiet"]
        )
        assert rc == 0

    def test_main_returns_nonzero_on_partial_failure(
        self, tmp_path: Path, monkeypatch, capsys,
    ):
        # Patch every loader to raise.
        monkeypatch.setattr(
            "rpps.metrics.compute_all.load_series",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("nope")),
        )
        monkeypatch.setattr(
            "rpps.metrics.compute_all.load_spliced_wages",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("nope")),
        )
        monkeypatch.setattr(
            "rpps.metrics.compute_all.basket_cost_panel",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("nope")),
        )

        rc = compute_all.main(["--output", str(tmp_path), "--quiet"])
        assert rc != 0


# ---------------------------------------------------------------------------
# Print-summary formatting
# ---------------------------------------------------------------------------

class TestPrintSummary:
    """Cover the human-readable summary emitter (lines 182-204)."""

    def test_main_without_quiet_prints_human_readable_summary(
        self, tmp_path: Path, patched_run, capsys,
    ):
        rc = compute_all.main(["--output", str(tmp_path)])
        captured = capsys.readouterr()
        assert rc == 0
        # Header banner appears.
        assert "rpps.metrics.compute_all" in captured.out
        # Each metric's status row appears.
        for tag in ("RPPH", "WICR", "PRWDI"):
            assert tag in captured.out
        # OK rows include observation counts.
        assert "obs" in captured.out
        # Footer with run-summary path.
        assert "Run summary:" in captured.out

    def test_print_summary_renders_error_rows(
        self, tmp_path: Path, monkeypatch, capsys,
    ):
        # Inject a single-metric failure and verify the ERROR row prints
        # the error message, not "obs".
        def fake_load_series(series_id: str, *args, **kwargs):
            if series_id == "CPIAUCNS":
                raise RuntimeError("simulated outage")
            if series_id == "OPHNFB":
                return _synthetic_productivity()
            if series_id == "COMPRNFB":
                return _synthetic_real_compensation()
            raise KeyError(series_id)

        monkeypatch.setattr(
            "rpps.metrics.compute_all.load_series", fake_load_series,
        )
        monkeypatch.setattr(
            "rpps.metrics.compute_all.load_spliced_wages",
            lambda *a, **kw: _synthetic_wages(),
        )
        monkeypatch.setattr(
            "rpps.metrics.compute_all.basket_cost_panel",
            lambda *a, **kw: _synthetic_basket_panel(),
        )

        rc = compute_all.main(["--output", str(tmp_path)])
        captured = capsys.readouterr()
        # Failure → nonzero exit.
        assert rc != 0
        # ERROR appears for the failing metric.
        assert "ERROR" in captured.out
        # OK appears for the surviving metrics.
        assert "OK" in captured.out
        # The propagated error string is shown.
        assert "simulated outage" in captured.out

    def test_print_summary_handles_missing_coverage_dates(
        self, tmp_path: Path, capsys,
    ):
        # Synthesize a summary dict that has 'ok' rows but no coverage dates,
        # exercising the conditional branch in _print_summary.
        summary = {
            "output_dir": str(tmp_path),
            "frequency": "M",
            "base_year": 1947,
            "results": [
                {"metric": "RPPH", "status": "ok", "n_observations": 100},
                {"metric": "WICR", "status": "error", "error": "boom"},
            ],
            "run_summary_path": str(tmp_path / "summary.json"),
        }
        compute_all._print_summary(summary)
        captured = capsys.readouterr()
        # "obs" still printed even without coverage dates.
        assert "100 obs" in captured.out
        # Error message rendered for the failing row.
        assert "boom" in captured.out


# ---------------------------------------------------------------------------
# Run-summary version tracking
# ---------------------------------------------------------------------------

class TestVersionTracking:
    """Verify the run summary records the actual installed package version
    rather than a hardcoded literal (regression for the 0.3.0 → 0.3.1 bug)."""

    def test_summary_records_current_package_version(
        self, tmp_path: Path, patched_run,
    ):
        from rpps import __version__ as expected
        summary = compute_all.run(output_dir=tmp_path)
        assert summary["rpps_version"] == expected

    def test_version_persisted_to_disk(
        self, tmp_path: Path, patched_run,
    ):
        from rpps import __version__ as expected
        compute_all.run(output_dir=tmp_path)
        with open(tmp_path / "metrics_run_summary.json") as fh:
            on_disk = json.load(fh)
        assert on_disk["rpps_version"] == expected


# ---------------------------------------------------------------------------
# Module-as-script entry point
# ---------------------------------------------------------------------------

class TestModuleEntryPoint:
    """The `if __name__ == "__main__"` guard is exercised by running the
    module as a script in a subprocess. We can't cover this line in-process
    (coverage.py considers it dead from `import` paths), but we can verify
    the script invocation works."""

    def test_module_runs_as_script(self, tmp_path: Path):
        import os
        import subprocess
        import sys

        # PYTHONIOENCODING + explicit UTF-8 decode so the test is portable
        # to Windows hosts whose default console codec is cp1252.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        # Run with --help to avoid needing any data; still exercises argparse.
        result = subprocess.run(
            [sys.executable, "-m", "rpps.metrics.compute_all", "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=env,
        )
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}; stderr was:\n{result.stderr}"
        )
        assert "rpps.metrics.compute_all" in result.stdout
        assert "--output" in result.stdout
        assert "--frequency" in result.stdout
        assert "--base-year" in result.stdout
        assert "--quiet" in result.stdout
