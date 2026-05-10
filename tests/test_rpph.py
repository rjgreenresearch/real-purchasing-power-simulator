"""Tests for rpps.metrics.rpph — Real Purchasing Power Hours."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rpps.metrics.rpph import (
    RpphResult,
    compute_rpph,
    labor_hours_for_item,
    save_rpph_result,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_panel_and_wage():
    """A 5-period basket panel with two items and a wage series."""
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    panel = pd.DataFrame(
        {
            "gasoline": [30.0, 30.0, 30.0, 30.0, 30.0],   # $30/period
            "beef":     [200.0, 200.0, 200.0, 200.0, 200.0],  # $200/period
        },
        index=idx,
    )
    wage = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=idx, name="AHETPI")
    return panel, wage


@pytest.fixture
def panel_with_gap():
    """Panel where one item has missing data in some periods."""
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    panel = pd.DataFrame(
        {
            "always_present": [100.0, 100.0, 100.0, 100.0, 100.0],
            "sparse":         [50.0, np.nan, 50.0, np.nan, 50.0],
        },
        index=idx,
    )
    wage = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=idx, name="AHETPI")
    return panel, wage


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------

class TestComputeRpph:
    def test_returns_rpph_result(self, simple_panel_and_wage):
        panel, wage = simple_panel_and_wage
        result = compute_rpph(panel, wage)
        assert isinstance(result, RpphResult)

    def test_per_item_hours_correct(self, simple_panel_and_wage):
        panel, wage = simple_panel_and_wage
        result = compute_rpph(panel, wage)
        # gasoline = 30 / 10 = 3 hours; beef = 200 / 10 = 20 hours.
        assert np.allclose(result.by_item["gasoline"].values, 3.0)
        assert np.allclose(result.by_item["beef"].values, 20.0)

    def test_composite_is_sum(self, simple_panel_and_wage):
        panel, wage = simple_panel_and_wage
        result = compute_rpph(panel, wage)
        # 3 + 20 = 23 hours per period.
        assert np.allclose(result.composite.values, 23.0)

    def test_n_observations(self, simple_panel_and_wage):
        panel, wage = simple_panel_and_wage
        result = compute_rpph(panel, wage)
        assert result.n_observations == 5

    def test_coverage_dates(self, simple_panel_and_wage):
        panel, wage = simple_panel_and_wage
        result = compute_rpph(panel, wage)
        assert result.coverage_start == panel.index[0]
        assert result.coverage_end == panel.index[-1]


class TestNaNHandling:
    def test_composite_nan_when_item_missing(self, panel_with_gap):
        panel, wage = panel_with_gap
        result = compute_rpph(panel, wage)
        # Composite should be NaN where 'sparse' is NaN.
        assert result.composite.isna().sum() == 2
        assert result.composite.notna().sum() == 3

    def test_per_item_preserves_nans(self, panel_with_gap):
        panel, wage = panel_with_gap
        result = compute_rpph(panel, wage)
        assert result.by_item["sparse"].isna().sum() == 2
        assert result.by_item["always_present"].isna().sum() == 0

    def test_require_all_items_raises_on_gap(self, panel_with_gap):
        panel, wage = panel_with_gap
        with pytest.raises(ValueError, match="missing values"):
            compute_rpph(panel, wage, require_all_items=True)

    def test_require_all_items_passes_when_complete(self, simple_panel_and_wage):
        panel, wage = simple_panel_and_wage
        result = compute_rpph(panel, wage, require_all_items=True)
        assert result.n_observations == 5


# ---------------------------------------------------------------------------
# Wage scaling — the metric's economic content
# ---------------------------------------------------------------------------

class TestWageScaling:
    def test_doubling_wage_halves_hours(self, simple_panel_and_wage):
        panel, wage = simple_panel_and_wage
        result_low = compute_rpph(panel, wage)
        result_high = compute_rpph(panel, wage * 2.0)
        ratio = result_high.composite / result_low.composite
        assert np.allclose(ratio.values, 0.5)

    def test_doubling_prices_doubles_hours(self, simple_panel_and_wage):
        panel, wage = simple_panel_and_wage
        result_baseline = compute_rpph(panel, wage)
        result_inflated = compute_rpph(panel * 2.0, wage)
        ratio = result_inflated.composite / result_baseline.composite
        assert np.allclose(ratio.values, 2.0)

    def test_proportional_inflation_neutral(self, simple_panel_and_wage):
        panel, wage = simple_panel_and_wage
        # If wages and prices both double, RPPH is unchanged.
        result = compute_rpph(panel * 2.0, wage * 2.0)
        baseline = compute_rpph(panel, wage)
        assert np.allclose(result.composite.values, baseline.composite.values)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_panel_raises(self):
        panel = pd.DataFrame()
        wage = pd.Series([10.0], index=pd.date_range("2020-01-31", periods=1, freq="ME"))
        with pytest.raises(ValueError, match="empty"):
            compute_rpph(panel, wage)

    def test_empty_wage_raises(self):
        idx = pd.date_range("2020-01-31", periods=2, freq="ME")
        panel = pd.DataFrame({"x": [1.0, 1.0]}, index=idx)
        with pytest.raises(ValueError, match="empty"):
            compute_rpph(panel, pd.Series([], dtype=float))

    def test_misaligned_indices_use_ffill(self):
        # Wage has fewer dates; should forward-fill into panel dates.
        panel_idx = pd.date_range("2020-01-31", periods=5, freq="ME")
        wage_idx = pd.DatetimeIndex(["2020-01-31", "2020-04-30"])
        panel = pd.DataFrame({"x": [10.0] * 5}, index=panel_idx)
        wage = pd.Series([5.0, 5.0], index=wage_idx, name="w")
        result = compute_rpph(panel, wage)
        assert np.allclose(result.composite.values, 2.0)


# ---------------------------------------------------------------------------
# Audit and serialization
# ---------------------------------------------------------------------------

class TestAuditAndSerialization:
    def test_audit_contains_expected_keys(self, simple_panel_and_wage):
        panel, wage = simple_panel_and_wage
        result = compute_rpph(panel, wage)
        for key in [
            "computation",
            "items",
            "wage_series_id",
            "panel_n_rows",
            "wage_n_rows",
            "composite_n_obs",
            "computed_at_utc",
        ]:
            assert key in result.audit

    def test_to_dict_is_json_serializable(self, simple_panel_and_wage):
        panel, wage = simple_panel_and_wage
        result = compute_rpph(panel, wage)
        json.dumps(result.to_dict(), default=str)

    def test_save_roundtrip(self, simple_panel_and_wage, tmp_path):
        panel, wage = simple_panel_and_wage
        result = compute_rpph(panel, wage)
        paths = save_rpph_result(result, tmp_path)
        for key in ["composite", "by_item", "audit"]:
            assert paths[key].exists()
        # Audit should be valid JSON.
        with open(paths["audit"]) as fh:
            doc = json.load(fh)
        assert doc["wage_series_id"] == "AHETPI"


# ---------------------------------------------------------------------------
# labor_hours_for_item convenience
# ---------------------------------------------------------------------------

class TestLaborHoursForItem:
    def test_basic(self):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        cost = pd.Series([100.0, 200.0, 400.0], index=idx, name="home")
        wage = pd.Series([10.0, 10.0, 10.0], index=idx)
        hours = labor_hours_for_item(cost, wage)
        assert np.allclose(hours.values, [10.0, 20.0, 40.0])
        assert hours.name == "home_hours"
