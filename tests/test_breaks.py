"""Tests for rpps.breaks — structural break detection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rpps.breaks import (
    BreakResult,
    detect_breaks_baiperron,
    quandt_andrews_test,
    save_break_result,
)


RNG = np.random.default_rng(20260508)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_regime_data():
    """100 quarterly obs, two variables, single break at index 50."""
    n = 100
    idx = pd.date_range("2000-03-31", periods=n, freq="QE")
    rng = np.random.default_rng(123)

    # Variable 1: mean 0 → mean 5 at index 50.
    v1 = np.concatenate([
        rng.normal(0.0, 0.5, 50),
        rng.normal(5.0, 0.5, 50),
    ])
    # Variable 2: mean 1 → mean -2 at index 50.
    v2 = np.concatenate([
        rng.normal(1.0, 0.5, 50),
        rng.normal(-2.0, 0.5, 50),
    ])
    return pd.DataFrame({"x1": v1, "x2": v2}, index=idx)


@pytest.fixture
def three_regime_data():
    """100 obs, two variables, breaks at 33 and 67."""
    n = 99
    idx = pd.date_range("2000-03-31", periods=n, freq="QE")
    rng = np.random.default_rng(456)
    v1 = np.concatenate([
        rng.normal(0.0, 0.4, 33),
        rng.normal(3.0, 0.4, 34),
        rng.normal(-1.0, 0.4, 32),
    ])
    v2 = np.concatenate([
        rng.normal(2.0, 0.4, 33),
        rng.normal(0.0, 0.4, 34),
        rng.normal(4.0, 0.4, 32),
    ])
    return pd.DataFrame({"x1": v1, "x2": v2}, index=idx)


@pytest.fixture
def stationary_data():
    """80 obs of stationary noise, no breaks."""
    n = 80
    idx = pd.date_range("2000-03-31", periods=n, freq="QE")
    rng = np.random.default_rng(789)
    return pd.DataFrame({
        "x1": rng.normal(0.0, 0.3, n),
        "x2": rng.normal(1.0, 0.3, n),
    }, index=idx)


# ---------------------------------------------------------------------------
# Bai-Perron (PELT) tests
# ---------------------------------------------------------------------------

class TestBaiPerronBasic:
    def test_returns_break_result(self, two_regime_data):
        result = detect_breaks_baiperron(two_regime_data)
        assert isinstance(result, BreakResult)

    def test_detects_single_break_in_two_regime_data(self, two_regime_data):
        result = detect_breaks_baiperron(two_regime_data)
        assert result.n_breaks >= 1
        # The detected break should be near index 50 (date around end of regime 1).
        actual_break = two_regime_data.index[50]
        if result.break_dates:
            time_diffs = [abs((d - actual_break).days) for d in result.break_dates]
            min_diff = min(time_diffs)
            # Detected break should be within 12 quarters (~3 years) of the true break.
            assert min_diff < 1100  # ~3 years in days

    def test_three_regime_data_finds_at_least_two_breaks(self, three_regime_data):
        result = detect_breaks_baiperron(three_regime_data)
        assert result.n_breaks >= 2
        assert result.n_regimes >= 3

    def test_regime_assignments_length(self, two_regime_data):
        result = detect_breaks_baiperron(two_regime_data)
        assert len(result.regime_assignments) == len(two_regime_data)

    def test_regime_assignments_monotonic(self, two_regime_data):
        result = detect_breaks_baiperron(two_regime_data)
        # Regime labels should weakly increase along the timeline.
        labels = result.regime_assignments.values
        diffs = np.diff(labels)
        assert (diffs >= 0).all()


class TestBaiPerronRegimeStructure:
    def test_regime_summary_has_one_row_per_regime(self, three_regime_data):
        result = detect_breaks_baiperron(three_regime_data)
        assert len(result.regime_summary) == result.n_regimes

    def test_regime_summary_contains_per_variable_stats(self, two_regime_data):
        result = detect_breaks_baiperron(two_regime_data)
        for col in ["regime_id", "start_date", "end_date", "n_observations",
                    "x1_mean", "x1_std", "x2_mean", "x2_std"]:
            assert col in result.regime_summary.columns

    def test_regime_means_differ_meaningfully(self, two_regime_data):
        result = detect_breaks_baiperron(two_regime_data)
        if result.n_regimes >= 2:
            means = result.regime_summary["x1_mean"].values
            # Adjacent regimes should have visually different means.
            assert abs(means[1] - means[0]) > 1.0


class TestBaiPerronEdgeCases:
    def test_empty_data_raises(self):
        df = pd.DataFrame({"x": []})
        df.index = pd.DatetimeIndex([])
        with pytest.raises(ValueError, match="empty"):
            detect_breaks_baiperron(df)

    def test_nan_data_raises(self, two_regime_data):
        data = two_regime_data.copy()
        data.iloc[10, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            detect_breaks_baiperron(data)

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(TypeError, match="DatetimeIndex"):
            detect_breaks_baiperron(df)


class TestBaiPerronAuditAndPersistence:
    def test_audit_keys_present(self, two_regime_data):
        result = detect_breaks_baiperron(two_regime_data)
        for key in ["method", "cost", "trim", "min_size", "penalty",
                    "n_observations", "n_variables", "computed_at_utc"]:
            assert key in result.audit

    def test_to_dict_json_serializable(self, two_regime_data):
        result = detect_breaks_baiperron(two_regime_data)
        json.dumps(result.to_dict(), default=str)

    def test_save_roundtrip(self, two_regime_data, tmp_path):
        result = detect_breaks_baiperron(two_regime_data)
        paths = save_break_result(result, tmp_path)
        for key in ["regimes", "summary", "audit"]:
            assert paths[key].exists()


# ---------------------------------------------------------------------------
# Quandt-Andrews tests
# ---------------------------------------------------------------------------

class TestQuandtAndrews:
    def test_returns_expected_keys(self, two_regime_data):
        idx = two_regime_data.index
        # Construct a regression-friendly DGP with a break in the slope.
        rng = np.random.default_rng(100)
        x = rng.normal(0.0, 1.0, len(idx))
        # Slope = 1.0 before break, 4.0 after.
        slope = np.where(np.arange(len(idx)) < 50, 1.0, 4.0)
        y = slope * x + rng.normal(0.0, 0.3, len(idx))
        y_s = pd.Series(y, index=idx)
        X = pd.DataFrame({"x": x}, index=idx)

        result = quandt_andrews_test(y_s, X)
        for key in ["sup_wald", "sup_wald_date", "p_value",
                    "candidate_dates", "wald_statistics"]:
            assert key in result

    def test_detects_known_break(self):
        # Generate data with a strong slope change at index 60.
        n = 120
        idx = pd.date_range("2000-03-31", periods=n, freq="QE")
        rng = np.random.default_rng(200)
        x = rng.normal(0.0, 1.0, n)
        slope = np.where(np.arange(n) < 60, 1.0, 5.0)
        y = slope * x + rng.normal(0.0, 0.2, n)
        y_s = pd.Series(y, index=idx)
        X = pd.DataFrame({"x": x}, index=idx)

        result = quandt_andrews_test(y_s, X, trim=0.15)
        # sup-Wald should be large, p-value small.
        assert result["sup_wald"] > 10
        # Detected break date should be within ±10 quarters of true.
        true_break = idx[60]
        days_diff = abs((result["sup_wald_date"] - true_break).days)
        assert days_diff < 1100  # ~3 years

    def test_no_break_yields_modest_sup_wald(self):
        n = 100
        idx = pd.date_range("2000-03-31", periods=n, freq="QE")
        rng = np.random.default_rng(300)
        x = rng.normal(0.0, 1.0, n)
        y = 1.5 * x + rng.normal(0.0, 0.3, n)
        result = quandt_andrews_test(
            pd.Series(y, index=idx),
            pd.DataFrame({"x": x}, index=idx),
            trim=0.15,
        )
        assert result["p_value"] > 0.05

    def test_small_sample_raises(self):
        n = 20
        idx = pd.date_range("2000-03-31", periods=n, freq="QE")
        with pytest.raises(ValueError, match="too small"):
            quandt_andrews_test(
                pd.Series(np.zeros(n), index=idx),
                pd.DataFrame({"x": np.zeros(n)}, index=idx),
            )
