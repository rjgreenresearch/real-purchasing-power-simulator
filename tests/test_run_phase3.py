"""Tests for the run_phase3 driver script.

The Phase 3 modules themselves (rpps.breaks, rpps.regression,
rpps.counterfactual) have their own test suites covering the actual
computations. These tests cover the driver's orchestration: argument
parsing, input validation, error handling, and the contract that the
driver writes the file paths the report module expects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the repo-root run_phase3 importable as a module.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import run_phase3  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_minimal_processed_dir(tmp_path: Path) -> Path:
    """Create a `processed/` directory with the inputs run_phase3 reads.

    The series are synthetic but match the schema and date ranges of real
    `make data` output, so the driver's load logic can be exercised.
    """
    p = tmp_path / "processed"
    p.mkdir()

    # Spliced wages: 1939-2025 monthly (post-1939 only, fine for breaks panel
    # which starts at 1947).
    idx_m = pd.date_range("1939-01-01", "2025-12-01", freq="MS")
    wages = pd.Series(np.linspace(0.40, 30.0, len(idx_m)), index=idx_m, name="spliced_wage")
    wages.to_csv(p / "spliced_wages.csv", header=True)

    # Spliced productivity: annual 1947-2025
    idx_y = pd.DatetimeIndex([f"{y}-12-31" for y in range(1947, 2026)])
    prod = pd.Series(np.linspace(50.0, 119.0, len(idx_y)), index=idx_y, name="spliced_prod")
    prod.to_csv(p / "spliced_productivity.csv", header=True)

    # rpph_by_item with at least a 'housing' column starting in 1963.
    idx_m_h = pd.date_range("1963-01-01", "2025-12-01", freq="MS")
    pd.DataFrame({
        "housing": np.linspace(7000, 13000, len(idx_m_h)),
    }, index=idx_m_h).to_csv(p / "rpph_by_item.csv")

    return p


def _build_minimal_fred_cache(tmp_path: Path) -> Path:
    """Create a fake FRED cache with the series the driver loads.

    The driver calls load_series for CPIAUCNS, PPIACO, OPHNFB, COMPRNFB.
    Each needs to be present at the expected path with the right CSV schema.
    """
    cache = tmp_path / "fred_cache"
    cache.mkdir()

    def write(series_id: str, freq: str, start: str, end: str):
        if freq == "M":
            idx = pd.date_range(start, end, freq="MS")
        else:  # Q
            idx = pd.date_range(start, end, freq="QS")
        # Smooth synthetic level path that admits YoY pct_change.
        n = len(idx)
        values = np.linspace(20.0, 320.0, n)
        df = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "value": values})
        df.to_csv(cache / f"{series_id}.csv", index=False)
        manifest = {
            "series_id": series_id,
            "frequency": freq,
            "n_observations": n,
            "first_date": str(idx[0].date()),
            "last_date": str(idx[-1].date()),
        }
        (cache / f"{series_id}.meta.json").write_text(json.dumps(manifest))

    write("CPIAUCNS", "M", "1947-01-01", "2025-12-01")
    write("PPIACO", "M", "1947-01-01", "2025-12-01")
    write("OPHNFB", "Q", "1947-01-01", "2025-10-01")
    write("COMPRNFB", "Q", "1947-01-01", "2025-10-01")
    return cache


# ---------------------------------------------------------------------------
# Argument parsing and validation
# ---------------------------------------------------------------------------

class TestArgumentParsing:
    def test_help_returns_zero_and_describes_steps(self, capsys):
        with pytest.raises(SystemExit) as exc:
            run_phase3.main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "rpps.breaks" in out
        assert "rpps.regression" in out
        assert "rpps.counterfactual" in out

    def test_missing_processed_dir_exits_nonzero(self, tmp_path, capsys):
        rc = run_phase3.main([
            "--processed-dir", str(tmp_path / "nonexistent"),
            "--skip-counterfactual", "--skip-regression", "--skip-breaks",
        ])
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err

    def test_missing_required_inputs_exits_nonzero(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        rc = run_phase3.main([
            "--processed-dir", str(empty),
            "--skip-counterfactual", "--skip-regression", "--skip-breaks",
        ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "spliced_wages.csv" in err

    def test_skip_breaks_without_existing_regimes_csv(self, tmp_path, capsys):
        p = _build_minimal_processed_dir(tmp_path)
        rc = run_phase3.main([
            "--processed-dir", str(p),
            "--skip-breaks",
            "--skip-regression",
            "--skip-counterfactual",
        ])
        assert rc == 1
        assert "breaks_regimes.csv" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Step contracts (output files match what report.load_inputs reads)
# ---------------------------------------------------------------------------

class TestStepContracts:
    """The driver must write filenames that exactly match what
    rpps.report.load_inputs looks for. If these tests fail, `make report`
    will silently skip the Phase 3 panel."""

    def test_break_save_filenames(self):
        # Inspect the constants in the save helper to lock in the filename
        # contract with the report module.
        import inspect

        from rpps import breaks as breaks_mod
        src = inspect.getsource(breaks_mod.save_break_result)
        assert "{prefix}_regimes.csv" in src
        assert "{prefix}_summary.csv" in src
        assert "{prefix}_audit.json" in src

    def test_regression_save_filenames(self):
        import inspect

        from rpps import regression as reg_mod
        src = inspect.getsource(reg_mod.save_regression_result)
        assert "{prefix}_audit.json" in src
        assert "{prefix}_cross_regime.csv" in src
        assert "{prefix}_by_regime_coefs.csv" in src

    def test_counterfactual_save_filenames(self):
        import inspect

        from rpps import counterfactual as cf_mod
        src = inspect.getsource(cf_mod.save_counterfactual_result)
        assert "{prefix}_panel.csv" in src
        assert "{prefix}_audit.json" in src

    def test_report_load_inputs_reads_phase3_filenames(self):
        # The flip side: report.load_inputs must look for these exact names.
        import inspect

        from rpps import report as rpt_mod
        src = inspect.getsource(rpt_mod.load_inputs)
        assert "breaks_regimes.csv" in src
        assert "breaks_summary.csv" in src
        assert "regression_cross_regime.csv" in src
        assert "regression_by_regime_coefs.csv" in src
        assert "counterfactual_panel.csv" in src


# ---------------------------------------------------------------------------
# step_breaks: panel construction with synthetic FRED cache
# ---------------------------------------------------------------------------

class TestStepBreaks:
    def test_panel_construction_starts_1947_and_has_four_columns(
        self, tmp_path, monkeypatch,
    ):
        p = _build_minimal_processed_dir(tmp_path)
        cache = _build_minimal_fred_cache(tmp_path)

        # Run only step_breaks; assert the saved files match contract.
        result_info = run_phase3.step_breaks(p, cache_dir=cache)
        panel = result_info["panel"]
        assert panel.shape[1] == 4
        assert set(panel.columns) == {"cpi_yoy", "ppi_yoy", "prod_yoy", "wage_yoy"}
        assert panel.index.min() >= pd.Timestamp("1947-01-01")
        # Save artifacts written
        assert (p / "breaks_regimes.csv").is_file()
        assert (p / "breaks_summary.csv").is_file()
        assert (p / "breaks_audit.json").is_file()


# ---------------------------------------------------------------------------
# step_regression: works against a synthetic regime assignment
# ---------------------------------------------------------------------------

class TestStepRegression:
    def test_regression_writes_outputs(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        cache = _build_minimal_fred_cache(tmp_path)

        # Build a synthetic two-regime assignment that aligns with the housing
        # RPPH index range.
        housing_idx = pd.date_range("1963-01-01", "2025-12-01", freq="MS")
        regimes = pd.Series(0, index=housing_idx, dtype=int)
        regimes.loc["1990-01-01":] = 1

        result_info = run_phase3.step_regression(p, cache_dir=cache, regime_assignments=regimes)
        assert result_info["n_regimes_fitted"] >= 1
        assert (p / "regression_audit.json").is_file()


# ---------------------------------------------------------------------------
# step_counterfactual
# ---------------------------------------------------------------------------

class TestStepCounterfactual:
    def test_counterfactual_writes_outputs(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        cache = _build_minimal_fred_cache(tmp_path)

        result_info = run_phase3.step_counterfactual(p, cache_dir=cache, n_bootstrap=50)
        assert "final_pct_gap" in result_info
        assert (p / "counterfactual_panel.csv").is_file()
        assert (p / "counterfactual_audit.json").is_file()


# ---------------------------------------------------------------------------
# End-to-end: skipping all steps still validates inputs
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_skip_all_with_existing_breaks_csv_succeeds(self, tmp_path, capsys):
        """If breaks_regimes.csv exists and all steps skipped, exit cleanly."""
        p = _build_minimal_processed_dir(tmp_path)
        # Pre-populate breaks_regimes.csv so --skip-breaks is satisfied.
        idx = pd.date_range("1947-01-01", "2025-12-01", freq="QS")
        regimes = pd.Series(0, index=idx, name="regime", dtype=int)
        regimes.to_csv(p / "breaks_regimes.csv", header=True)

        rc = run_phase3.main([
            "--processed-dir", str(p),
            "--skip-breaks",
            "--skip-regression",
            "--skip-counterfactual",
        ])
        assert rc == 0
        assert "Phase 3 complete" in capsys.readouterr().out
