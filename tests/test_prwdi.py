"""Tests for rpps.metrics.prwdi — Productivity-Real-Wage Decoupling Index."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rpps.metrics.prwdi import (
    DEFAULT_BASE_YEAR,
    PrwdiResult,
    compute_prwdi,
    save_prwdi_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_quarterly_series(start_year: int, n_years: int, annual_growth: float, initial: float) -> pd.Series:
    n = n_years * 4
    idx = pd.date_range(f"{start_year}-03-31", periods=n, freq="QE")
    quarterly_factor = (1.0 + annual_growth) ** (1.0 / 4.0)
    values = initial * np.power(quarterly_factor, np.arange(n))
    return pd.Series(values, index=idx)


@pytest.fixture
def coupled_series():
    """Productivity and compensation grow at the same rate. PRWDI should be ~1.0."""
    prod = _build_quarterly_series(1947, 25, 0.025, 100.0)
    comp = _build_quarterly_series(1947, 25, 0.025, 100.0)
    return prod, comp


@pytest.fixture
def decoupled_series():
    """Productivity grows 3% YoY, compensation 1% YoY → strong decoupling."""
    prod = _build_quarterly_series(1947, 50, 0.03, 100.0)
    comp = _build_quarterly_series(1947, 50, 0.01, 100.0)
    return prod, comp


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------

class TestBasic:
    def test_returns_prwdi_result(self, coupled_series):
        prod, comp = coupled_series
        result = compute_prwdi(prod, comp, base_year=1947)
        assert isinstance(result, PrwdiResult)

    def test_default_base_year(self):
        assert DEFAULT_BASE_YEAR == 1947

    def test_prwdi_unity_at_base_year(self, coupled_series):
        prod, comp = coupled_series
        result = compute_prwdi(prod, comp, base_year=1947)
        # Mean of base-year PRWDI values should be very close to 1.
        in_base = result.prwdi[result.prwdi.index.year == 1947]
        assert np.allclose(in_base.mean(), 1.0, atol=1e-6)

    def test_coupled_series_yields_unity_throughout(self, coupled_series):
        prod, comp = coupled_series
        result = compute_prwdi(prod, comp, base_year=1947)
        # Equal growth rates → PRWDI should be 1.0 everywhere.
        assert np.allclose(result.prwdi.values, 1.0, atol=1e-6)


class TestDecoupling:
    def test_prwdi_rises_when_productivity_outpaces_compensation(self, decoupled_series):
        prod, comp = decoupled_series
        result = compute_prwdi(prod, comp, base_year=1947)
        # 50 years of 2pp differential. PRWDI(1997) ≈ (1.03/1.01)**50 ≈ 2.66.
        end_prwdi = result.prwdi.dropna().iloc[-1]
        assert end_prwdi > 2.5
        assert end_prwdi < 3.0

    def test_prwdi_below_unity_when_compensation_outpaces_productivity(self):
        prod = _build_quarterly_series(1947, 30, 0.01, 100.0)
        comp = _build_quarterly_series(1947, 30, 0.03, 100.0)
        result = compute_prwdi(prod, comp, base_year=1947)
        end_prwdi = result.prwdi.dropna().iloc[-1]
        assert end_prwdi < 1.0

    def test_delta_prwdi_positive_under_decoupling(self, decoupled_series):
        prod, comp = decoupled_series
        result = compute_prwdi(prod, comp, base_year=1947)
        # Under sustained decoupling, the year-over-year change in PRWDI is positive.
        steady = result.delta_prwdi_annual.dropna().iloc[20:]
        assert (steady > 0).mean() > 0.95


# ---------------------------------------------------------------------------
# Base year handling
# ---------------------------------------------------------------------------

class TestBaseYear:
    def test_alternative_base_year(self, coupled_series):
        prod, comp = coupled_series
        result = compute_prwdi(prod, comp, base_year=1960)
        in_base = result.prwdi[result.prwdi.index.year == 1960]
        assert np.allclose(in_base.mean(), 1.0, atol=1e-6)

    def test_missing_base_year_raises(self):
        # Series only covers 1990-2000; ask for 1947 base.
        prod = _build_quarterly_series(1990, 10, 0.02, 100.0)
        comp = _build_quarterly_series(1990, 10, 0.02, 100.0)
        with pytest.raises(ValueError, match="base_year"):
            compute_prwdi(prod, comp, base_year=1947)

    def test_audit_records_base_values(self, coupled_series):
        prod, comp = coupled_series
        result = compute_prwdi(prod, comp, base_year=1947)
        assert "productivity_base_value" in result.audit
        assert "compensation_base_value" in result.audit
        assert result.audit["productivity_base_value"] > 0


# ---------------------------------------------------------------------------
# Annual frequency support
# ---------------------------------------------------------------------------

class TestAnnualFrequency:
    def test_annual_data_handled(self):
        idx = pd.DatetimeIndex([f"{y}-12-31" for y in range(1925, 1958)])
        prod = pd.Series(
            [100.0 * (1.02 ** (y - 1925)) for y in range(1925, 1958)], index=idx
        )
        comp = pd.Series(
            [100.0 * (1.015 ** (y - 1925)) for y in range(1925, 1958)], index=idx
        )
        result = compute_prwdi(prod, comp, base_year=1929)
        assert result.audit["annual_lag_periods"] == 1
        assert result.n_observations == 33


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_productivity_raises(self):
        prod = pd.Series([], dtype=float)
        comp = _build_quarterly_series(1947, 5, 0.02, 100.0)
        with pytest.raises(ValueError, match="productivity"):
            compute_prwdi(prod, comp, base_year=1947)

    def test_empty_compensation_raises(self):
        prod = _build_quarterly_series(1947, 5, 0.02, 100.0)
        comp = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="real_compensation"):
            compute_prwdi(prod, comp, base_year=1947)

    def test_no_overlap_raises(self):
        prod = _build_quarterly_series(1947, 5, 0.02, 100.0)
        comp = _build_quarterly_series(2000, 5, 0.02, 100.0)
        # Both have base years available, but no temporal overlap.
        with pytest.raises(ValueError):
            compute_prwdi(prod, comp, base_year=1947)


# ---------------------------------------------------------------------------
# Audit and serialization
# ---------------------------------------------------------------------------

class TestAuditAndSerialization:
    def test_audit_keys(self, coupled_series):
        prod, comp = coupled_series
        result = compute_prwdi(prod, comp, base_year=1947)
        for key in [
            "computation",
            "base_year",
            "productivity_base_value",
            "compensation_base_value",
            "joint_n_obs",
            "computed_at_utc",
        ]:
            assert key in result.audit

    def test_to_dict_json_serializable(self, coupled_series):
        prod, comp = coupled_series
        result = compute_prwdi(prod, comp, base_year=1947)
        json.dumps(result.to_dict(), default=str)

    def test_save_roundtrip(self, coupled_series, tmp_path):
        prod, comp = coupled_series
        result = compute_prwdi(prod, comp, base_year=1947)
        paths = save_prwdi_result(result, tmp_path)
        assert paths["panel"].exists()
        assert paths["audit"].exists()
        df = pd.read_csv(paths["panel"], index_col=0, parse_dates=True)
        assert "prwdi" in df.columns
        assert "productivity_index" in df.columns
        assert "compensation_index" in df.columns
