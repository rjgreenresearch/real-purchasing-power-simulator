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

    # wicr_panel.csv with the high_wicr_run boolean. We synthesize a pattern
    # that matches the empirically observed structure: True in 1973-1981 and
    # 2021-2024, False elsewhere. This gives variation across the synthetic
    # two-regime split used by TestStepH3 and TestStepRegression.
    idx_m_w = pd.date_range("1947-01-01", "2025-12-01", freq="MS")
    high_wicr_run = pd.Series(False, index=idx_m_w, dtype=bool)
    high_wicr_run.loc["1973-01-01":"1981-12-01"] = True
    high_wicr_run.loc["2021-01-01":"2024-12-01"] = True
    pd.DataFrame({
        "wage_growth_yoy": np.linspace(0.05, 0.04, len(idx_m_w)),
        "inflation_yoy": np.linspace(0.03, 0.03, len(idx_m_w)),
        "wicr_yoy": np.linspace(0.6, 0.75, len(idx_m_w)),
        "wicr_smoothed": np.linspace(0.6, 0.75, len(idx_m_w)),
        "regime_label": ["medium"] * len(idx_m_w),
        "high_wicr_run": high_wicr_run,
    }, index=idx_m_w).to_csv(p / "wicr_panel.csv")

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
# step_h3
# ---------------------------------------------------------------------------

class TestStepH3:
    def test_h3_writes_outputs_with_expected_filenames(self, tmp_path):
        """File-naming contract: H3 outputs go to regression_h3_*."""
        p = _build_minimal_processed_dir(tmp_path)
        cache = _build_minimal_fred_cache(tmp_path)

        housing_idx = pd.date_range("1963-01-01", "2025-12-01", freq="MS")
        regimes = pd.Series(0, index=housing_idx, dtype=int)
        regimes.loc["1990-01-01":] = 1

        result_info = run_phase3.step_h3(p, cache_dir=cache, regime_assignments=regimes)
        assert result_info["n_regimes_fitted"] >= 1
        assert (p / "regression_h3_audit.json").is_file()
        assert (p / "regression_h3_by_regime_coefs.csv").is_file()
        assert (p / "regression_h3_cross_regime.csv").is_file()

    def test_h3_audit_includes_specification(self, tmp_path):
        p = _build_minimal_processed_dir(tmp_path)
        cache = _build_minimal_fred_cache(tmp_path)

        housing_idx = pd.date_range("1963-01-01", "2025-12-01", freq="MS")
        regimes = pd.Series(0, index=housing_idx, dtype=int)
        regimes.loc["1990-01-01":] = 1

        run_phase3.step_h3(p, cache_dir=cache, regime_assignments=regimes)
        audit = json.loads((p / "regression_h3_audit.json").read_text())
        # The audit dict comes from RegimeRegressionResult.to_dict(); the
        # H3-specific keys are nested under audit["audit"].
        assert "h3_specification" in audit["audit"]
        assert audit["audit"]["h3_target_coefficient"] == "dlog_wage_x_high_wicr"
        assert "h3_variation_diagnostic" in audit["audit"]

    def test_h3_target_coefficient_is_interaction(self, tmp_path):
        """The cross-regime tests should be on the interaction column."""
        p = _build_minimal_processed_dir(tmp_path)
        cache = _build_minimal_fred_cache(tmp_path)

        housing_idx = pd.date_range("1963-01-01", "2025-12-01", freq="MS")
        regimes = pd.Series(0, index=housing_idx, dtype=int)
        regimes.loc["1990-01-01":] = 1

        result_info = run_phase3.step_h3(p, cache_dir=cache, regime_assignments=regimes)
        result = result_info["result"]
        assert result.target_coefficient == "dlog_wage_x_high_wicr"
        # Each by_regime fit should have the interaction term as a regressor
        # (even if rank-deficient → NaN coefficients).
        for regime_id, res in result.by_regime.items():
            assert "dlog_wage_x_high_wicr" in res.coefficients.index, \
                f"regime {regime_id} missing interaction coefficient"

    def test_h3_variation_diagnostic_flags_no_variation(self, tmp_path):
        """If a regime has no high_wicr variation, the diagnostic flags it."""
        p = _build_minimal_processed_dir(tmp_path)
        cache = _build_minimal_fred_cache(tmp_path)

        # Build an assignment where regime 0 is entirely 1947-1972 (no
        # high-WICR overlap given our fixture pattern, since the fixture
        # turns high_wicr_run on at 1973), and regime 1 is everything else.
        # Regime 0's quarterly panel will have all-False high_wicr → no
        # variation in the dummy.
        housing_idx = pd.date_range("1963-01-01", "2025-12-01", freq="MS")
        regimes = pd.Series(0, index=housing_idx, dtype=int)
        regimes.loc["1973-01-01":] = 1

        result_info = run_phase3.step_h3(p, cache_dir=cache, regime_assignments=regimes)
        diag = result_info["variation_diag"]
        # Regime 0 (1963-1972) has no high-WICR months → no dummy variation
        assert diag[0]["dummy_varies"] is False
        assert diag[0]["n_high_wicr"] == 0
        # Regime 1 (1973-2025) has both high and low → variation
        assert diag[1]["dummy_varies"] is True
        assert diag[1]["n_high_wicr"] > 0
        assert diag[1]["n_low_wicr"] > 0

    def test_h3_missing_wicr_column_raises(self, tmp_path):
        """Removing high_wicr_run from wicr_panel.csv should raise a clear error."""
        p = _build_minimal_processed_dir(tmp_path)
        cache = _build_minimal_fred_cache(tmp_path)

        # Strip the high_wicr_run column.
        wicr_path = p / "wicr_panel.csv"
        wicr_df = pd.read_csv(wicr_path, index_col=0, parse_dates=True)
        wicr_df.drop(columns=["high_wicr_run"]).to_csv(wicr_path)

        housing_idx = pd.date_range("1963-01-01", "2025-12-01", freq="MS")
        regimes = pd.Series(0, index=housing_idx, dtype=int)
        regimes.loc["1990-01-01":] = 1

        with pytest.raises(RuntimeError, match="high_wicr_run"):
            run_phase3.step_h3(p, cache_dir=cache, regime_assignments=regimes)

    def test_h3_skipped_via_cli_flag(self, tmp_path, capsys):
        """--skip-h3 should not produce regression_h3_audit.json."""
        p = _build_minimal_processed_dir(tmp_path)
        # Pre-populate breaks_regimes.csv so --skip-breaks is satisfied.
        idx = pd.date_range("1947-01-01", "2025-12-01", freq="QS")
        regimes = pd.Series(0, index=idx, name="regime", dtype=int)
        regimes.to_csv(p / "breaks_regimes.csv", header=True)

        rc = run_phase3.main([
            "--processed-dir", str(p),
            "--skip-breaks",
            "--skip-regression",
            "--skip-h3",
            "--skip-counterfactual",
        ])
        assert rc == 0
        assert not (p / "regression_h3_audit.json").is_file()

    def test_h3_auto_skipped_when_wicr_panel_missing(self, tmp_path, capsys):
        """If wicr_panel.csv doesn't exist, the H3 step warns and skips."""
        p = _build_minimal_processed_dir(tmp_path)
        # Remove wicr_panel.csv to simulate user forgetting `make metrics`.
        (p / "wicr_panel.csv").unlink()

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
        # H3 was auto-skipped; no audit file produced.
        assert not (p / "regression_h3_audit.json").is_file()


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
            "--skip-h3",
            "--skip-counterfactual",
        ])
        assert rc == 0
        assert "Phase 3 complete" in capsys.readouterr().out
