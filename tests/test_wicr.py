"""Tests for rpps.metrics.wicr — Wage-Inflation Capture Ratio."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rpps.metrics.wicr import (
    WICR_HIGH_THRESHOLD,
    WICR_LOW_THRESHOLD,
    WicrResult,
    _flag_sustained_runs,
    compute_wicr,
    save_wicr_result,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_constant_growth_series(
    start: str,
    n_months: int,
    annual_growth: float,
    initial: float = 100.0,
) -> pd.Series:
    """Series with constant compound annual growth, monthly granularity."""
    idx = pd.date_range(start, periods=n_months, freq="ME")
    monthly_factor = (1.0 + annual_growth) ** (1.0 / 12.0)
    values = initial * np.power(monthly_factor, np.arange(n_months))
    return pd.Series(values, index=idx)


@pytest.fixture
def stable_4pct_wages_2pct_inflation():
    """36 months. Wages grow 4% YoY, prices grow 2% YoY. WICR ≈ 0.50."""
    wage = _build_constant_growth_series("1990-01-31", 60, 0.04)
    cpi = _build_constant_growth_series("1990-01-31", 60, 0.02)
    return wage, cpi


@pytest.fixture
def high_wicr_period():
    """36 months. Wages 5% YoY, inflation 4.5% YoY. WICR ≈ 0.90."""
    wage = _build_constant_growth_series("2020-01-31", 60, 0.05)
    cpi = _build_constant_growth_series("2020-01-31", 60, 0.045)
    return wage, cpi


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------

class TestBasic:
    def test_returns_wicr_result(self, stable_4pct_wages_2pct_inflation):
        wage, cpi = stable_4pct_wages_2pct_inflation
        result = compute_wicr(wage, cpi)
        assert isinstance(result, WicrResult)

    def test_yoy_lag_for_monthly_data_is_12(self, stable_4pct_wages_2pct_inflation):
        wage, cpi = stable_4pct_wages_2pct_inflation
        result = compute_wicr(wage, cpi)
        assert result.audit["yoy_lag_periods"] == 12

    def test_wicr_value_for_known_growth_pair(self, stable_4pct_wages_2pct_inflation):
        wage, cpi = stable_4pct_wages_2pct_inflation
        result = compute_wicr(wage, cpi)
        # Once past the first 12-month YoY warmup, smoothed WICR ≈ 0.02 / 0.04 = 0.50.
        steady = result.wicr_smoothed.dropna().iloc[-12:]
        assert np.allclose(steady.values, 0.50, atol=0.01)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

class TestRegimeLabel:
    def test_low_wicr_regime_for_low_capture(self, stable_4pct_wages_2pct_inflation):
        wage, cpi = stable_4pct_wages_2pct_inflation
        result = compute_wicr(wage, cpi)
        # WICR ~ 0.50: should be "low" or boundary "medium".
        steady_label = result.regime_label.dropna().iloc[-1]
        assert steady_label in ("low", "medium")

    def test_high_wicr_regime_for_high_capture(self, high_wicr_period):
        wage, cpi = high_wicr_period
        result = compute_wicr(wage, cpi)
        # WICR ~ 0.90: should be "high".
        steady_label = result.regime_label.dropna().iloc[-1]
        assert steady_label == "high"

    def test_threshold_constants(self):
        assert WICR_LOW_THRESHOLD == 0.50
        assert WICR_HIGH_THRESHOLD == 0.80


# ---------------------------------------------------------------------------
# Sustained-run detection
# ---------------------------------------------------------------------------

class TestSustainedRuns:
    def test_runs_below_min_are_not_flagged(self):
        s = pd.Series([False, True, True, False, True, True, False])
        result = _flag_sustained_runs(s, min_length=3)
        assert not result.any()

    def test_runs_at_min_are_flagged(self):
        s = pd.Series([False, True, True, True, False, True, True])
        result = _flag_sustained_runs(s, min_length=3)
        # First run of length 3 flagged; second of length 2 not.
        expected = [False, True, True, True, False, False, False]
        for i, want in enumerate(expected):
            assert bool(result.iloc[i]) is want, f"index {i}: got {result.iloc[i]}, want {want}"

    def test_long_run_fully_flagged(self):
        s = pd.Series([True] * 10)
        result = _flag_sustained_runs(s, min_length=3)
        assert result.all()

    def test_high_wicr_period_flags_sustained_run(self, high_wicr_period):
        wage, cpi = high_wicr_period
        result = compute_wicr(wage, cpi, sustained_periods=8)
        # 60 months of stable high WICR should produce many sustained-run periods.
        assert result.high_wicr_runs.sum() > 20

    def test_stable_low_wicr_period_no_sustained_runs(self, stable_4pct_wages_2pct_inflation):
        wage, cpi = stable_4pct_wages_2pct_inflation
        result = compute_wicr(wage, cpi)
        assert result.high_wicr_runs.sum() == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_wage_raises(self):
        wage = pd.Series([], dtype=float)
        cpi = _build_constant_growth_series("1990-01-31", 24, 0.02)
        with pytest.raises(ValueError, match="wage_series is empty"):
            compute_wicr(wage, cpi)

    def test_insufficient_overlap_raises(self):
        wage = _build_constant_growth_series("1990-01-31", 6, 0.04)
        cpi = _build_constant_growth_series("1990-01-31", 6, 0.02)
        with pytest.raises(ValueError, match="Insufficient overlap"):
            compute_wicr(wage, cpi)

    def test_zero_wage_growth_yields_nan_wicr(self):
        # Wage flat, CPI rising → infinite WICR. Implementation masks near-zero
        # denominators to NaN.
        idx = pd.date_range("1990-01-31", periods=24, freq="ME")
        wage = pd.Series([10.0] * 24, index=idx)
        cpi = _build_constant_growth_series("1990-01-31", 24, 0.02)
        result = compute_wicr(wage, cpi)
        # The post-warmup WICR values should be NaN due to the eps mask.
        post_warmup = result.wicr_yoy.iloc[12:]
        assert post_warmup.isna().sum() >= 8


# ---------------------------------------------------------------------------
# Audit and serialization
# ---------------------------------------------------------------------------

class TestAuditAndSerialization:
    def test_audit_keys_present(self, stable_4pct_wages_2pct_inflation):
        wage, cpi = stable_4pct_wages_2pct_inflation
        result = compute_wicr(wage, cpi)
        expected = [
            "computation",
            "yoy_lag_periods",
            "target_frequency",
            "smoothing_periods",
            "low_threshold",
            "high_threshold",
            "sustained_periods",
            "joint_n_obs",
            "computed_at_utc",
        ]
        for k in expected:
            assert k in result.audit

    def test_to_dict_json_serializable(self, stable_4pct_wages_2pct_inflation):
        wage, cpi = stable_4pct_wages_2pct_inflation
        result = compute_wicr(wage, cpi)
        json.dumps(result.to_dict(), default=str)

    def test_save_roundtrip(self, stable_4pct_wages_2pct_inflation, tmp_path):
        wage, cpi = stable_4pct_wages_2pct_inflation
        result = compute_wicr(wage, cpi)
        paths = save_wicr_result(result, tmp_path)
        assert paths["panel"].exists()
        assert paths["audit"].exists()
        # Round-trip the panel.
        df = pd.read_csv(paths["panel"], index_col=0, parse_dates=True)
        assert "wicr_yoy" in df.columns
        assert "wicr_smoothed" in df.columns
        assert "regime_label" in df.columns


# ---------------------------------------------------------------------------
# Helper-function edge cases
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    """Both empty-input branches should raise. Previously only wages was
    covered; this closes the cpi_series-empty branch (line 153)."""

    def test_empty_cpi_raises(self):
        wage = _build_constant_growth_series("1990-01-31", 24, 0.04)
        cpi = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="cpi_series is empty"):
            compute_wicr(wage, cpi)


class TestDatetimeIndexCoercion:
    """Inputs with a non-DatetimeIndex (e.g. plain int labels representing
    years) should be coerced internally rather than rejected."""

    def test_non_datetime_index_is_coerced(self):
        # Build wage and CPI with a string-date index that pandas can parse,
        # but is not yet a DatetimeIndex. compute_wicr should accept this.
        n = 60
        date_strings = pd.date_range("1990-01-31", periods=n, freq="ME").strftime("%Y-%m-%d")
        wage_values = (1.04 ** (1.0 / 12.0)) ** np.arange(n) * 10.0
        cpi_values = (1.02 ** (1.0 / 12.0)) ** np.arange(n) * 100.0
        wage = pd.Series(wage_values, index=pd.Index(list(date_strings)))
        cpi = pd.Series(cpi_values, index=pd.Index(list(date_strings)))
        # Should not raise; coercion happens inside _to_datetime_index.
        result = compute_wicr(wage, cpi)
        assert result.n_observations > 0


class TestFrequencyDetection:
    """Cover quarterly, annual, and fallback branches of _detect_freq
    (lines 249-262)."""

    def test_quarterly_data_yoy_lag_is_4(self):
        # 80 quarterly periods, no irregular spacing.
        idx = pd.date_range("1990-03-31", periods=80, freq="QE")
        wage = pd.Series((1.04 ** (1.0 / 4.0)) ** np.arange(80) * 10.0, index=idx)
        cpi = pd.Series((1.02 ** (1.0 / 4.0)) ** np.arange(80) * 100.0, index=idx)
        result = compute_wicr(wage, cpi)
        assert result.audit["yoy_lag_periods"] == 4
        assert result.audit["target_frequency"] == "Q"

    def test_irregular_spacing_falls_back_to_median_diff(self):
        # Construct an irregular monthly-ish index (mostly 30-day spacing
        # with occasional gaps). pd.infer_freq returns None; the fallback
        # branch counts median day-spacing.
        from rpps.metrics.wicr import _detect_freq

        base = pd.Timestamp("1990-01-01")
        # 60 dates spaced ~30 days but with a couple of jitters
        dates = [base + pd.Timedelta(days=30 * i + (3 if i % 5 == 0 else 0))
                 for i in range(60)]
        idx = pd.DatetimeIndex(dates)
        # Should be classified as monthly (median ~30 days).
        assert _detect_freq(idx) == "M"

    def test_irregular_quarterly_spacing_falls_back_correctly(self):
        from rpps.metrics.wicr import _detect_freq

        base = pd.Timestamp("1990-01-01")
        # 30 dates spaced ~91 days with jitter — quarterly-like.
        dates = [base + pd.Timedelta(days=91 * i + (i % 7))
                 for i in range(30)]
        idx = pd.DatetimeIndex(dates)
        assert _detect_freq(idx) == "Q"

    def test_irregular_annual_spacing_falls_back_correctly(self):
        from rpps.metrics.wicr import _detect_freq

        base = pd.Timestamp("1900-01-01")
        # 50 dates spaced ~365 days with jitter — annual-like.
        dates = [base + pd.Timedelta(days=365 * i + (i % 4))
                 for i in range(50)]
        idx = pd.DatetimeIndex(dates)
        assert _detect_freq(idx) == "A"

    def test_single_observation_index_defaults_to_monthly(self):
        from rpps.metrics.wicr import _detect_freq

        idx = pd.DatetimeIndex([pd.Timestamp("2020-01-01")])
        # With <2 observations, the fallback default is "M".
        assert _detect_freq(idx) == "M"
