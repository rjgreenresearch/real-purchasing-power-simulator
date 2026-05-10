"""Tests for rpps.regression — within-regime regression with HAC SEs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rpps.regression import (
    RegimeRegressionResult,
    RegressionResult,
    fit_by_regime,
    fit_ols_hac,
    save_regression_result,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def known_dgp():
    """Linear DGP with known coefficients: y = 1 + 2*x1 - 0.5*x2 + ε."""
    n = 200
    idx = pd.date_range("2000-03-31", periods=n, freq="QE")
    rng = np.random.default_rng(42)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(0.0, 0.5, n)
    return pd.Series(y, index=idx, name="y"), pd.DataFrame({"x1": x1, "x2": x2}, index=idx)


@pytest.fixture
def regime_dgp():
    """DGP with two regimes, different slopes."""
    n = 240
    idx = pd.date_range("2000-03-31", periods=n, freq="QE")
    rng = np.random.default_rng(123)
    x = rng.normal(0.0, 1.0, n)
    # Regime 0 (first 120): slope = 2.0. Regime 1: slope = 0.5.
    slope = np.where(np.arange(n) < 120, 2.0, 0.5)
    y = slope * x + rng.normal(0.0, 0.4, n)
    regime = pd.Series(np.where(np.arange(n) < 120, 0, 1), index=idx, name="regime")
    return (
        pd.Series(y, index=idx, name="y"),
        pd.DataFrame({"x": x}, index=idx),
        regime,
    )


# ---------------------------------------------------------------------------
# Single regression
# ---------------------------------------------------------------------------

class TestFitOlsHac:
    def test_returns_regression_result(self, known_dgp):
        y, X = known_dgp
        result = fit_ols_hac(y, X)
        assert isinstance(result, RegressionResult)

    def test_recovers_known_coefficients(self, known_dgp):
        y, X = known_dgp
        result = fit_ols_hac(y, X)
        assert abs(float(result.coefficients["const"]) - 1.0) < 0.10
        assert abs(float(result.coefficients["x1"]) - 2.0) < 0.10
        assert abs(float(result.coefficients["x2"]) - (-0.5)) < 0.10

    def test_n_observations_matches(self, known_dgp):
        y, X = known_dgp
        result = fit_ols_hac(y, X)
        assert result.n_observations == 200

    def test_hac_lag_default_uses_newey_west_rule(self, known_dgp):
        y, X = known_dgp
        result = fit_ols_hac(y, X)
        # Newey-West (1994): L = floor(4 * (200/100)^(2/9)) = floor(4 * 1.166) = 4
        assert result.hac_lag == 4

    def test_explicit_hac_lag(self, known_dgp):
        y, X = known_dgp
        result = fit_ols_hac(y, X, hac_lag=8)
        assert result.hac_lag == 8

    def test_p_values_correct_signs(self, known_dgp):
        y, X = known_dgp
        result = fit_ols_hac(y, X)
        # All true coefficients are nonzero, with n=200 should be highly significant.
        assert result.p_values["x1"] < 0.001
        assert result.p_values["x2"] < 0.05

    def test_r_squared_in_unit_interval(self, known_dgp):
        y, X = known_dgp
        result = fit_ols_hac(y, X)
        assert 0.0 <= result.r_squared <= 1.0
        assert result.adj_r_squared <= result.r_squared

    def test_residuals_have_sample_mean_near_zero(self, known_dgp):
        y, X = known_dgp
        result = fit_ols_hac(y, X)
        assert abs(float(result.residuals.mean())) < 0.05

    def test_coefficient_table_shape(self, known_dgp):
        y, X = known_dgp
        result = fit_ols_hac(y, X)
        table = result.coefficient_table()
        assert list(table.columns) == ["coef", "std_err", "t_stat", "p_value"]
        assert len(table) == 3  # const + 2 regressors


class TestFitOlsHacEdgeCases:
    def test_empty_y_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fit_ols_hac(
                pd.Series([], dtype=float),
                pd.DataFrame({"x": []}),
            )

    def test_length_mismatch_raises(self):
        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        with pytest.raises(ValueError, match="same length"):
            fit_ols_hac(
                pd.Series([1.0] * 10, index=idx),
                pd.DataFrame({"x": [1.0] * 9}, index=idx[:9]),
            )

    def test_nan_in_y_raises(self):
        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        y = pd.Series([1.0, np.nan] + [3.0] * 8, index=idx)
        X = pd.DataFrame({"x": [1.0] * 10}, index=idx)
        with pytest.raises(ValueError, match="NaN"):
            fit_ols_hac(y, X)


# ---------------------------------------------------------------------------
# Regime-stratified regression
# ---------------------------------------------------------------------------

class TestFitByRegime:
    def test_returns_regime_regression_result(self, regime_dgp):
        y, X, regime = regime_dgp
        result = fit_by_regime(y, X, regime, target_coefficient="x")
        assert isinstance(result, RegimeRegressionResult)

    def test_one_result_per_regime(self, regime_dgp):
        y, X, regime = regime_dgp
        result = fit_by_regime(y, X, regime, target_coefficient="x")
        assert len(result.by_regime) == 2

    def test_recovers_regime_specific_slopes(self, regime_dgp):
        y, X, regime = regime_dgp
        result = fit_by_regime(y, X, regime, target_coefficient="x")
        beta_0 = float(result.by_regime[0].coefficients["x"])
        beta_1 = float(result.by_regime[1].coefficients["x"])
        assert abs(beta_0 - 2.0) < 0.10
        assert abs(beta_1 - 0.5) < 0.10

    def test_cross_regime_test_detects_difference(self, regime_dgp):
        y, X, regime = regime_dgp
        result = fit_by_regime(y, X, regime, target_coefficient="x")
        # The slopes are 2.0 and 0.5 — easily significantly different.
        tests = result.cross_regime_tests
        assert len(tests) == 1
        row = tests.iloc[0]
        assert row["regime_a"] == 0
        assert row["regime_b"] == 1
        assert row["p_value"] < 0.001

    def test_undersized_regimes_skipped(self):
        n = 120
        idx = pd.date_range("2000-03-31", periods=n, freq="QE")
        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 1.0, n)
        y = 1.5 * x + rng.normal(0.0, 0.3, n)
        # Three regimes; middle is tiny (only 5 obs).
        regime = pd.Series(
            [0] * 60 + [1] * 5 + [2] * 55,
            index=idx,
        )
        result = fit_by_regime(
            pd.Series(y, index=idx),
            pd.DataFrame({"x": x}, index=idx),
            regime,
            target_coefficient="x",
            min_regime_n=20,
        )
        assert 1 in result.audit["skipped_regimes"]
        assert len(result.by_regime) == 2

    def test_pairwise_count(self):
        # 3 regimes → 3 pairwise comparisons.
        n = 240
        idx = pd.date_range("2000-03-31", periods=n, freq="QE")
        rng = np.random.default_rng(8)
        x = rng.normal(0.0, 1.0, n)
        slopes = np.array([1.0] * 80 + [2.0] * 80 + [3.0] * 80)
        y = slopes * x + rng.normal(0.0, 0.3, n)
        regime = pd.Series(
            [0] * 80 + [1] * 80 + [2] * 80,
            index=idx,
        )
        result = fit_by_regime(
            pd.Series(y, index=idx),
            pd.DataFrame({"x": x}, index=idx),
            regime,
            target_coefficient="x",
        )
        assert len(result.cross_regime_tests) == 3


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_single_regression(self, known_dgp, tmp_path):
        y, X = known_dgp
        result = fit_ols_hac(y, X)
        paths = save_regression_result(result, tmp_path)
        assert paths["audit"].exists()

    def test_save_regime_regression_includes_csvs(self, regime_dgp, tmp_path):
        y, X, regime = regime_dgp
        result = fit_by_regime(y, X, regime, target_coefficient="x")
        paths = save_regression_result(result, tmp_path)
        assert paths["audit"].exists()
        assert paths["cross_regime"].exists()
        assert paths["by_regime_coefs"].exists()

    def test_audit_json_serializable(self, regime_dgp):
        y, X, regime = regime_dgp
        result = fit_by_regime(y, X, regime, target_coefficient="x")
        json.dumps(result.to_dict(), default=str)
