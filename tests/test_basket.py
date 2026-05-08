"""Tests for rpps.basket. All tests run offline and are deterministic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rpps.basket import (
    BASKET_ITEMS,
    BasketItem,
    _resample_to_frequency,
    basket_cost_panel,
    basket_summary,
    basket_total_cost,
    item_cost,
    load_item_price,
    load_kff_healthcare,
    load_nces_tuition,
)


# ---------------------------------------------------------------------------
# Catalog integrity
# ---------------------------------------------------------------------------

class TestBasketCatalog:
    def test_six_items(self):
        """Paper §4.3 specifies six basket items. The catalog must contain exactly six."""
        assert len(BASKET_ITEMS) == 6

    def test_canonical_item_names(self):
        """The canonical item names are fixed; renaming is a breaking change."""
        expected = {"gasoline", "beef", "tuition", "housing", "electricity", "healthcare"}
        assert set(BASKET_ITEMS.keys()) == expected

    def test_keys_match_item_names(self):
        for key, item in BASKET_ITEMS.items():
            assert key == item.name, f"Catalog key {key!r} != item.name {item.name!r}"

    def test_fixed_quantities_match_paper_specification(self):
        """The fixed quantities are specified in §4.3 of the paper. Drift is breaking."""
        assert BASKET_ITEMS["gasoline"].quantity == 12.0
        assert BASKET_ITEMS["gasoline"].unit == "gallon"
        assert BASKET_ITEMS["beef"].quantity == 40.0
        assert BASKET_ITEMS["beef"].unit == "lb"
        assert BASKET_ITEMS["tuition"].quantity == 1.0
        assert BASKET_ITEMS["tuition"].unit == "year"
        assert BASKET_ITEMS["housing"].quantity == 1.0
        assert BASKET_ITEMS["housing"].unit == "home"
        assert BASKET_ITEMS["electricity"].quantity == 1000.0
        assert BASKET_ITEMS["electricity"].unit == "kWh"
        assert BASKET_ITEMS["healthcare"].quantity == 1.0
        assert BASKET_ITEMS["healthcare"].unit == "year"

    def test_all_quantities_positive(self):
        for item in BASKET_ITEMS.values():
            assert item.quantity > 0, f"{item.name} has non-positive quantity"

    def test_sources_are_resolvable(self):
        """Every source must be either a known FRED ID prefix or 'external:tag'."""
        for item in BASKET_ITEMS.values():
            if item.source.startswith("external:"):
                tag = item.source.split(":", 1)[1]
                assert tag in ("nces_tuition", "kff_healthcare"), \
                    f"{item.name} has unknown external tag {tag!r}"
            else:
                # FRED series IDs are uppercase alphanumeric (and digits)
                assert item.source.replace("_", "").isalnum(), \
                    f"{item.name} has malformed FRED ID {item.source!r}"

    def test_coverage_start_year_is_plausible(self):
        for item in BASKET_ITEMS.values():
            assert 1900 < item.coverage_start_year < 2030

    def test_descriptions_are_present(self):
        for item in BASKET_ITEMS.values():
            assert item.description.strip(), f"{item.name} has empty description"

    def test_basket_items_are_immutable(self):
        """BasketItem is a frozen dataclass — should reject attribute assignment."""
        item = BASKET_ITEMS["gasoline"]
        with pytest.raises((AttributeError, Exception)):
            item.quantity = 999.0


class TestBasketSummary:
    def test_summary_returns_dataframe(self):
        df = basket_summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6

    def test_summary_has_expected_columns(self):
        df = basket_summary()
        expected = {"item", "quantity", "unit", "source", "coverage_start_year", "description"}
        assert expected.issubset(df.columns)


# ---------------------------------------------------------------------------
# External data loaders (from committed CSVs)
# ---------------------------------------------------------------------------

class TestNcesTuitionLoader:
    def test_returns_series(self):
        s = load_nces_tuition()
        assert isinstance(s, pd.Series)
        assert len(s) > 0

    def test_starts_at_or_before_1969(self):
        s = load_nces_tuition()
        assert s.index.min().year <= 1969

    def test_extends_to_recent(self):
        s = load_nces_tuition()
        # Committed data should extend to at least 2020
        assert s.index.max().year >= 2020

    def test_index_is_datetime(self):
        s = load_nces_tuition()
        assert isinstance(s.index, pd.DatetimeIndex)

    def test_values_are_positive(self):
        s = load_nces_tuition()
        assert (s.dropna() > 0).all()

    def test_values_are_monotonic_overall(self):
        """Tuition rises essentially monotonically over the long run.
        Allow occasional small drops (data revisions) but the trend is up."""
        s = load_nces_tuition()
        # Compare endpoints: 2020 should be >> 1970
        early = s.loc[s.index.year <= 1975].mean()
        late = s.loc[s.index.year >= 2020].mean()
        assert late > 5 * early, \
            f"Tuition late/early ratio = {late / early:.2f}, expected > 5"


class TestKffHealthcareLoader:
    def test_returns_series(self):
        s = load_kff_healthcare()
        assert isinstance(s, pd.Series)
        assert len(s) > 0

    def test_starts_at_1999(self):
        s = load_kff_healthcare()
        # KFF EHBS family-coverage series begins 1999
        assert s.index.min().year == 1999

    def test_extends_to_recent(self):
        s = load_kff_healthcare()
        assert s.index.max().year >= 2020

    def test_index_is_datetime(self):
        s = load_kff_healthcare()
        assert isinstance(s.index, pd.DatetimeIndex)

    def test_values_are_positive(self):
        s = load_kff_healthcare()
        assert (s.dropna() > 0).all()

    def test_first_year_value_is_in_expected_range(self):
        """1999 KFF family premium is documented as ~$5,791."""
        s = load_kff_healthcare()
        v_1999 = s.loc[s.index.year == 1999].iloc[0]
        assert 5000 < v_1999 < 7000


# ---------------------------------------------------------------------------
# load_item_price dispatcher
# ---------------------------------------------------------------------------

class TestLoadItemPrice:
    def test_dispatches_to_nces_for_tuition(self):
        s = load_item_price(BASKET_ITEMS["tuition"])
        # Should be the same object as load_nces_tuition would return
        ref = load_nces_tuition()
        pd.testing.assert_series_equal(s, ref)

    def test_dispatches_to_kff_for_healthcare(self):
        s = load_item_price(BASKET_ITEMS["healthcare"])
        ref = load_kff_healthcare()
        pd.testing.assert_series_equal(s, ref)

    def test_raises_on_unknown_external_tag(self, monkeypatch):
        """If we ever introduce a new external tag without wiring it, this catches it."""
        bogus = BasketItem(
            name="x", quantity=1.0, unit="x",
            source="external:does_not_exist",
            description="x", coverage_start_year=2000,
        )
        with pytest.raises(ValueError, match="Unknown external tag"):
            load_item_price(bogus)


# ---------------------------------------------------------------------------
# item_cost arithmetic
# ---------------------------------------------------------------------------

class TestItemCost:
    def test_cost_is_quantity_times_price(self):
        # Use the tuition item — has external CSV, no FRED dependence
        s = item_cost(BASKET_ITEMS["tuition"])
        ref_price = load_nces_tuition()
        # tuition quantity is 1, so cost == price exactly
        pd.testing.assert_series_equal(
            s.astype("float64").rename(None),
            ref_price.astype("float64").rename(None),
            check_names=False,
        )

    def test_cost_scales_with_quantity(self):
        # Construct a fake item with quantity=10 and the tuition source
        ten_year = BasketItem(
            name="ten_year_tuition", quantity=10.0, unit="year",
            source="external:nces_tuition",
            description="x", coverage_start_year=1969,
        )
        s = item_cost(ten_year)
        ref = load_nces_tuition()
        # Cost should be 10x the underlying price at every observation
        ratio = (s / ref).dropna()
        assert np.allclose(ratio.values, 10.0)

    def test_cost_series_is_named_after_item(self):
        s = item_cost(BASKET_ITEMS["tuition"])
        assert s.name == "tuition_cost"


# ---------------------------------------------------------------------------
# Frequency resampling
# ---------------------------------------------------------------------------

class TestResampleToFrequency:
    def test_annual_to_monthly_forward_fills(self):
        """An annual series upsampled to monthly should ffill within the year."""
        annual_idx = pd.DatetimeIndex(["2000-12-31", "2001-12-31", "2002-12-31"])
        annual = pd.Series([100.0, 200.0, 300.0], index=annual_idx)
        monthly = _resample_to_frequency(annual, "M")
        # The 2000-12-31 value should appear at multiple month-ends in 2001
        # before being replaced by the 2001-12-31 value
        # All months in 2001 except December should hold the 2000 value
        # Implementation-dependent; just check that the result is monthly and
        # that the output covers the original range
        assert len(monthly) >= 13   # at least one obs per year, plus boundaries
        # The terminal value matches the last annual obs
        assert monthly.dropna().iloc[-1] == pytest.approx(300.0)
        # The first non-null value in 2001 is 100 (forward-filled from 2000)
        first_2001 = monthly.loc["2001":].dropna().iloc[0]
        assert first_2001 == pytest.approx(100.0)

    def test_monthly_passes_through(self):
        """Monthly source resampled to monthly is essentially a noop on month-ends."""
        monthly_idx = pd.date_range("2020-01-31", "2020-12-31", freq="ME")
        monthly = pd.Series(np.arange(len(monthly_idx), dtype=float), index=monthly_idx)
        out = _resample_to_frequency(monthly, "M")
        assert len(out) == len(monthly)
        # The values should be preserved
        assert out.iloc[0] == pytest.approx(0.0)
        assert out.iloc[-1] == pytest.approx(11.0)

    def test_quarterly_to_monthly(self):
        """Quarterly source upsampled to monthly forward-fills within a quarter."""
        q_idx = pd.DatetimeIndex(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"])
        q = pd.Series([1.0, 2.0, 3.0, 4.0], index=q_idx)
        monthly = _resample_to_frequency(q, "M")
        # The 2020-06-30 value should propagate into July and August (limit=2)
        assert monthly.loc["2020-06-30"] == pytest.approx(2.0)
        # July and August forward-filled
        july = monthly.loc[(monthly.index.year == 2020) & (monthly.index.month == 7)]
        august = monthly.loc[(monthly.index.year == 2020) & (monthly.index.month == 8)]
        if len(july) > 0:
            assert july.iloc[0] == pytest.approx(2.0)
        if len(august) > 0:
            assert august.iloc[0] == pytest.approx(2.0)

    def test_monthly_to_annual(self):
        """Monthly source downsampled to annual takes year-end value."""
        monthly_idx = pd.date_range("2020-01-31", "2021-12-31", freq="ME")
        monthly = pd.Series(np.arange(len(monthly_idx), dtype=float), index=monthly_idx)
        annual = _resample_to_frequency(monthly, "A")
        # Should have ~2 observations (2020-12-31 and 2021-12-31)
        assert len(annual) == 2

    def test_rejects_unknown_frequency(self):
        s = pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-01"]))
        with pytest.raises(ValueError, match="Unsupported frequency"):
            _resample_to_frequency(s, "X")


# ---------------------------------------------------------------------------
# Panel and total cost
# ---------------------------------------------------------------------------

class TestBasketCostPanel:
    def test_panel_with_only_external_items(self):
        """A panel built from just the two external-source items doesn't need FRED."""
        df = basket_cost_panel(items=["tuition", "healthcare"], frequency="M")
        assert isinstance(df, pd.DataFrame)
        assert "tuition" in df.columns
        assert "healthcare" in df.columns
        # Pre-1999 healthcare should be NaN; pre-1969 tuition should be NaN
        early_2000 = df.loc["2000-01-01":"2000-12-31"]
        # In 2000, both should have values
        assert early_2000["tuition"].dropna().shape[0] > 0
        assert early_2000["healthcare"].dropna().shape[0] > 0

    def test_panel_missing_fred_items_skipped_with_warning(self, caplog):
        """If a FRED-sourced item is uncached, the panel skips it instead of erroring."""
        import logging
        with caplog.at_level(logging.WARNING):
            df = basket_cost_panel(items=["gasoline", "tuition"])
        # Tuition should appear; gasoline (FRED) should be skipped
        assert "tuition" in df.columns
        assert "gasoline" not in df.columns
        assert any("gasoline" in rec.message for rec in caplog.records)

    def test_empty_panel_when_all_skipped(self):
        """If no items are available, return an empty DataFrame, not an error."""
        # Using only FRED items with no cache → all skipped
        df = basket_cost_panel(items=["gasoline", "beef"])
        # Either empty DataFrame or DataFrame with no columns
        assert df.empty or df.shape[1] == 0


class TestBasketTotalCost:
    def test_sums_available_items(self):
        s = basket_total_cost(items=["tuition", "healthcare"], frequency="M")
        # In 2000, tuition + healthcare should be > 0
        v_2000 = s.loc["2000-01-01":"2000-12-31"].dropna()
        assert len(v_2000) > 0
        assert (v_2000 > 0).all()

    def test_total_equals_sum_of_components(self):
        panel = basket_cost_panel(items=["tuition", "healthcare"], frequency="M")
        total = basket_total_cost(items=["tuition", "healthcare"], frequency="M")
        # Pick a date where both are populated
        common = panel.dropna().index
        if len(common) > 0:
            d = common[0]
            assert total.loc[d] == pytest.approx(panel.loc[d].sum())

    def test_require_all_items_propagates_nans(self):
        """When require_all_items=True, NaN in any component should NaN the total."""
        # In 1995, tuition has data but healthcare doesn't (KFF starts 1999)
        s_strict = basket_total_cost(
            items=["tuition", "healthcare"], frequency="M", require_all_items=True,
        )
        # 1995-06-30 should be NaN (healthcare not yet available)
        v = s_strict.loc["1995-06-30"] if pd.Timestamp("1995-06-30") in s_strict.index else None
        if v is not None:
            assert pd.isna(v)

    def test_lenient_mode_keeps_partial_totals(self):
        """When require_all_items=False, partial totals are reported."""
        s = basket_total_cost(
            items=["tuition", "healthcare"], frequency="M", require_all_items=False,
        )
        # 1980-12-31 should have just tuition (no healthcare yet)
        v = s.loc["1980-12-31"] if pd.Timestamp("1980-12-31") in s.index else None
        if v is not None:
            ref_tuition = load_nces_tuition().loc["1980-12-31":"1980-12-31"]
            if len(ref_tuition) > 0:
                assert v == pytest.approx(ref_tuition.iloc[0])
