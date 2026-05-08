"""Tests for rpps.counterfactual — H4 counterfactual real-compensation gap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rpps.counterfactual import (
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_RANDOM_SEED,
    DEFAULT_REFERENCE_END,
    DEFAULT_REFERENCE_START,
    CounterfactualResult,
    _build_cf_series,
    _fit_reference,
    _last_date_in_year,
    compute_counterfactual,
    save_counterfactual_result,
)


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

def _quarterly_index(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """Quarterly DatetimeIndex (quarter-end) spanning [start_year, end_year]."""
    return pd.date_range(
        f"{start_year}-03-31",
        f"{end_year}-12-31",
        freq="QE",
    )


def _build_dgp(
    start_year: int = 1947,
    end_year: int = 2024,
    *,
    pre_alpha: float = 0.0,
    pre_beta: float = 1.0,
    post_alpha: float | None = None,
    post_beta: float | None = None,
    pivot_year: int = 1971,
    prod_q_growth: float = 0.005,  # ~2% annualized
    prod_q_shock_sd: float = 0.008,  # variation in Δlog(Q) per quarter
    noise_sd: float = 0.0,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Construct a productivity / compensation pair from a known DGP.

    Productivity Q follows a random walk with positive drift:
        Δlog(Q_t) = prod_q_growth + η_t,   η_t ~ N(0, prod_q_shock_sd)
    so that Δlog(Q) varies across periods (necessary to identify β).

    Compensation C is generated from Δlog(C) = α + β·Δlog(Q) + ε, with
    distinct (α, β) before and after `pivot_year` if `post_alpha`/`post_beta`
    are provided.

    Parameters
    ----------
    pre_alpha, pre_beta : floats
        Coefficients in the pre-pivot regime (the reference window).
    post_alpha, post_beta : floats or None
        Coefficients in the post-pivot regime. If None, identical to pre.
    pivot_year : int
        Year at which the regime switches.
    prod_q_growth : float
        Mean log-difference of productivity per period.
    prod_q_shock_sd : float
        Standard deviation of period-to-period productivity-growth shocks.
        Must be > 0 for β to be identified in the reference fit.
    noise_sd : float
        Standard deviation of additive Gaussian noise on Δlog(C).
    """
    if post_alpha is None:
        post_alpha = pre_alpha
    if post_beta is None:
        post_beta = pre_beta

    idx = _quarterly_index(start_year, end_year)
    n = len(idx)

    rng = np.random.default_rng(seed)

    # Productivity: random walk with drift so that Δlog(Q) varies.
    d_log_prod_innov = rng.normal(prod_q_growth, prod_q_shock_sd, size=n)
    d_log_prod_innov[0] = 0.0  # the first "difference" is undefined; set zero
    log_prod = np.log(100.0) + np.cumsum(d_log_prod_innov)
    prod_values = np.exp(log_prod)

    # Compensation: built up from Δlog(C) = α + β·Δlog(Q) + ε per period.
    log_comp = np.empty(n)
    log_comp[0] = np.log(100.0)
    for t in range(1, n):
        is_post = idx[t].year > pivot_year
        a = post_alpha if is_post else pre_alpha
        b = post_beta if is_post else pre_beta
        eps = rng.normal(0.0, noise_sd) if noise_sd > 0 else 0.0
        log_comp[t] = log_comp[t - 1] + a + b * d_log_prod_innov[t] + eps
    comp_values = np.exp(log_comp)

    prod = pd.Series(prod_values, index=idx, name="productivity")
    comp = pd.Series(comp_values, index=idx, name="real_compensation")
    return prod, comp


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_default_reference_window_is_1948_to_1971(self):
        # Per §5.4, the canonical reference window is the postwar shared-
        # prosperity period 1948-1971.
        assert DEFAULT_REFERENCE_START == 1948
        assert DEFAULT_REFERENCE_END == 1971

    def test_default_seed_is_42(self):
        # Hard-coded for reproducibility across the simulator.
        assert DEFAULT_RANDOM_SEED == 42

    def test_default_n_bootstrap_is_1000(self):
        assert DEFAULT_N_BOOTSTRAP == 1000


# ---------------------------------------------------------------------------
# Result-container contract
# ---------------------------------------------------------------------------

class TestResultContract:
    def test_returns_counterfactual_result(self):
        prod, comp = _build_dgp()
        result = compute_counterfactual(prod, comp, n_bootstrap=50)
        assert isinstance(result, CounterfactualResult)

    def test_required_attributes(self):
        prod, comp = _build_dgp()
        result = compute_counterfactual(prod, comp, n_bootstrap=50)
        for attr in (
            "counterfactual", "actual", "gap", "pct_gap", "final_pct_gap",
            "final_pct_gap_ci", "reference_alpha", "reference_beta",
            "reference_n", "n_bootstrap", "audit",
        ):
            assert hasattr(result, attr), f"missing attribute: {attr}"

    def test_ci_is_two_tuple_with_lo_le_hi(self):
        prod, comp = _build_dgp(noise_sd=0.005)
        result = compute_counterfactual(prod, comp, n_bootstrap=100)
        ci = result.final_pct_gap_ci
        assert isinstance(ci, tuple) and len(ci) == 2
        assert ci[0] <= ci[1]

    def test_audit_dict_has_required_keys(self):
        prod, comp = _build_dgp()
        result = compute_counterfactual(prod, comp, n_bootstrap=50)
        required = {
            "reference_window_years",
            "reference_alpha",
            "reference_beta",
            "reference_n_observations",
            "post_reference_n",
            "post_reference_start",
            "post_reference_end",
            "n_bootstrap",
            "random_seed",
            "confidence_level",
            "computed_at_utc",
        }
        assert required.issubset(set(result.audit.keys()))

    def test_to_dict_is_json_serializable(self):
        prod, comp = _build_dgp()
        result = compute_counterfactual(prod, comp, n_bootstrap=50)
        d = result.to_dict()
        json.dumps(d, default=str)

    def test_to_dict_contains_ci_bounds(self):
        prod, comp = _build_dgp(noise_sd=0.003)
        result = compute_counterfactual(prod, comp, n_bootstrap=80)
        d = result.to_dict()
        assert "final_pct_gap_ci_low" in d
        assert "final_pct_gap_ci_high" in d
        assert d["final_pct_gap_ci_low"] <= d["final_pct_gap_ci_high"]


# ---------------------------------------------------------------------------
# Reference-window coefficient recovery (DGP recovery)
# ---------------------------------------------------------------------------

class TestReferenceFit:
    def test_recovers_unit_beta_from_coupled_dgp(self):
        # No noise, β=1, α=0: OLS should recover (≈0, ≈1).
        prod, comp = _build_dgp(pre_alpha=0.0, pre_beta=1.0, noise_sd=0.0)
        result = compute_counterfactual(prod, comp, n_bootstrap=20)
        assert result.reference_beta == pytest.approx(1.0, abs=1e-6)
        assert result.reference_alpha == pytest.approx(0.0, abs=1e-6)

    def test_recovers_partial_beta_from_decoupled_dgp(self):
        # In reference window: β=0.7, α=0.001 per quarter, no noise.
        prod, comp = _build_dgp(
            pre_alpha=0.001, pre_beta=0.7, noise_sd=0.0,
        )
        result = compute_counterfactual(prod, comp, n_bootstrap=20)
        assert result.reference_beta == pytest.approx(0.7, abs=1e-6)
        assert result.reference_alpha == pytest.approx(0.001, abs=1e-6)

    def test_reference_n_matches_window(self):
        prod, comp = _build_dgp()
        result = compute_counterfactual(
            prod, comp,
            reference_start=1948,
            reference_end=1971,
            n_bootstrap=20,
        )
        # 24 years × 4 quarters = 96 levels → 95 differences in 1948Q2-1971Q4.
        # The first 1948 difference uses 1948Q1 as base, but the differencing
        # is computed on the in-window slice only — that masks 1947 levels —
        # so we expect ~95 differences.
        assert 90 <= result.reference_n <= 96


# ---------------------------------------------------------------------------
# Counterfactual construction (continuity, formula correctness)
# ---------------------------------------------------------------------------

class TestCounterfactualSeries:
    def test_pivot_value_equals_actual(self):
        prod, comp = _build_dgp()
        result = compute_counterfactual(prod, comp, n_bootstrap=20)
        # The first observation of the post-reference window is the pivot
        # date; counterfactual must start at the actual value.
        assert result.counterfactual.iloc[0] == pytest.approx(
            result.actual.iloc[0]
        )

    def test_coupled_dgp_yields_zero_gap(self):
        # If α=0, β=1 throughout, counterfactual ≡ actual → zero gap.
        prod, comp = _build_dgp(
            pre_alpha=0.0, pre_beta=1.0,
            post_alpha=0.0, post_beta=1.0,
            noise_sd=0.0,
        )
        result = compute_counterfactual(prod, comp, n_bootstrap=20)
        # Allow small numerical drift from compounding.
        assert abs(result.final_pct_gap) < 1e-6
        assert result.gap.abs().max() < 1e-6

    def test_decoupling_post_pivot_yields_positive_gap(self):
        # Pre-1971: β=1.0 (workers capture full productivity).
        # Post-1971: β=0.5 (workers capture half). Gap should be POSITIVE
        # because counterfactual reflects the more generous reference regime.
        prod, comp = _build_dgp(
            pre_alpha=0.0, pre_beta=1.0,
            post_alpha=0.0, post_beta=0.5,
            noise_sd=0.0,
        )
        result = compute_counterfactual(prod, comp, n_bootstrap=20)
        assert result.final_pct_gap > 0
        # Magnitude check: ~50 years post-1971 × ~0.5pp lower per-period
        # decoupling × growth → should be substantial.
        assert result.final_pct_gap > 0.05  # at least 5% gap

    def test_reverse_decoupling_yields_negative_gap(self):
        # Pre-1971: β=0.5; post-1971: β=1.0 (workers capture MORE post-pivot).
        # Counterfactual is below actual → negative gap.
        prod, comp = _build_dgp(
            pre_alpha=0.0, pre_beta=0.5,
            post_alpha=0.0, post_beta=1.0,
            noise_sd=0.0,
        )
        result = compute_counterfactual(prod, comp, n_bootstrap=20)
        assert result.final_pct_gap < 0

    def test_gap_equals_counterfactual_minus_actual(self):
        prod, comp = _build_dgp(
            pre_beta=1.0, post_beta=0.6, noise_sd=0.0,
        )
        result = compute_counterfactual(prod, comp, n_bootstrap=20)
        manual_gap = result.counterfactual - result.actual
        pd.testing.assert_series_equal(
            result.gap.rename(None),
            manual_gap.rename(None),
            check_names=False,
            atol=1e-12,
        )

    def test_pct_gap_equals_gap_over_actual(self):
        prod, comp = _build_dgp(
            pre_beta=1.0, post_beta=0.6, noise_sd=0.0,
        )
        result = compute_counterfactual(prod, comp, n_bootstrap=20)
        manual_pct = result.gap / result.actual
        pd.testing.assert_series_equal(
            result.pct_gap.rename(None),
            manual_pct.rename(None),
            check_names=False,
            atol=1e-12,
        )

    def test_final_pct_gap_matches_last_observation(self):
        prod, comp = _build_dgp(
            pre_beta=1.0, post_beta=0.5, noise_sd=0.0,
        )
        result = compute_counterfactual(prod, comp, n_bootstrap=20)
        assert result.final_pct_gap == pytest.approx(
            result.pct_gap.iloc[-1]
        )


# ---------------------------------------------------------------------------
# Bootstrap CI behavior
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_zero_noise_yields_tight_ci(self):
        prod, comp = _build_dgp(
            pre_beta=1.0, post_beta=0.5, noise_sd=0.0,
        )
        result = compute_counterfactual(prod, comp, n_bootstrap=200)
        # With zero residuals, every bootstrap iteration gives the same
        # estimate, so the CI collapses to a point.
        lo, hi = result.final_pct_gap_ci
        assert hi - lo < 1e-6

    def test_higher_noise_widens_ci(self):
        # Generate two datasets with the same regime change but different noise.
        prod_a, comp_a = _build_dgp(
            pre_beta=1.0, post_beta=0.5,
            noise_sd=0.001, seed=11,
        )
        prod_b, comp_b = _build_dgp(
            pre_beta=1.0, post_beta=0.5,
            noise_sd=0.010, seed=11,
        )
        result_a = compute_counterfactual(prod_a, comp_a, n_bootstrap=200)
        result_b = compute_counterfactual(prod_b, comp_b, n_bootstrap=200)
        width_a = result_a.final_pct_gap_ci[1] - result_a.final_pct_gap_ci[0]
        width_b = result_b.final_pct_gap_ci[1] - result_b.final_pct_gap_ci[0]
        assert width_b > width_a

    def test_seed_reproducibility(self):
        prod, comp = _build_dgp(
            pre_beta=1.0, post_beta=0.5, noise_sd=0.005, seed=7,
        )
        a = compute_counterfactual(prod, comp, n_bootstrap=100, random_seed=42)
        b = compute_counterfactual(prod, comp, n_bootstrap=100, random_seed=42)
        assert a.final_pct_gap_ci == b.final_pct_gap_ci

    def test_different_seed_changes_ci(self):
        prod, comp = _build_dgp(
            pre_beta=1.0, post_beta=0.5, noise_sd=0.005, seed=7,
        )
        a = compute_counterfactual(prod, comp, n_bootstrap=100, random_seed=42)
        b = compute_counterfactual(prod, comp, n_bootstrap=100, random_seed=43)
        # CI bounds should differ even slightly with a different RNG path.
        assert a.final_pct_gap_ci != b.final_pct_gap_ci

    def test_ci_brackets_point_estimate_for_smooth_dgp(self):
        prod, comp = _build_dgp(
            pre_beta=1.0, post_beta=0.5, noise_sd=0.003, seed=13,
        )
        result = compute_counterfactual(prod, comp, n_bootstrap=400)
        lo, hi = result.final_pct_gap_ci
        # The bootstrap re-fits the regression and re-builds the CF, so the
        # point estimate need not lie exactly inside the percentile interval —
        # but for a well-specified DGP with moderate n_bootstrap, the
        # estimate should fall reasonably close to the interval.
        assert lo - 0.05 < result.final_pct_gap < hi + 0.05

    def test_confidence_level_widens_ci(self):
        prod, comp = _build_dgp(
            pre_beta=1.0, post_beta=0.5, noise_sd=0.005, seed=7,
        )
        narrow = compute_counterfactual(
            prod, comp, n_bootstrap=200, confidence_level=0.50,
        )
        wide = compute_counterfactual(
            prod, comp, n_bootstrap=200, confidence_level=0.99,
        )
        narrow_w = narrow.final_pct_gap_ci[1] - narrow.final_pct_gap_ci[0]
        wide_w = wide.final_pct_gap_ci[1] - wide.final_pct_gap_ci[0]
        assert wide_w > narrow_w


# ---------------------------------------------------------------------------
# Reference-window parameter
# ---------------------------------------------------------------------------

class TestReferenceWindow:
    def test_audit_records_window(self):
        prod, comp = _build_dgp()
        result = compute_counterfactual(
            prod, comp,
            reference_start=1955,
            reference_end=1980,
            n_bootstrap=20,
        )
        assert result.audit["reference_window_years"] == [1955, 1980]

    def test_post_reference_start_after_window_end(self):
        prod, comp = _build_dgp()
        result = compute_counterfactual(
            prod, comp,
            reference_start=1948,
            reference_end=1971,
            n_bootstrap=20,
        )
        # The post-reference path begins at the last observation in 1971
        # (the pivot), which is the end of the reference window.
        post_start = pd.Timestamp(result.audit["post_reference_start"])
        assert post_start.year == 1971


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_last_date_in_year_returns_q4(self):
        idx = _quarterly_index(2000, 2002)
        s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
        d = _last_date_in_year(s, 2001)
        assert d is not None
        assert d.year == 2001
        assert d.month == 12

    def test_last_date_in_year_returns_none_when_absent(self):
        idx = _quarterly_index(2000, 2002)
        s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
        assert _last_date_in_year(s, 1980) is None

    def test_fit_reference_recovers_clean_coefficients(self):
        prod, comp = _build_dgp(
            pre_alpha=0.002, pre_beta=0.8, noise_sd=0.0,
        )
        alpha, beta, residuals, n = _fit_reference(
            prod, comp, 1948, 1971,
        )
        assert alpha == pytest.approx(0.002, abs=1e-6)
        assert beta == pytest.approx(0.8, abs=1e-6)
        assert n == len(residuals)
        # Zero-noise DGP: residuals must be ~0.
        assert float(np.abs(residuals).max()) < 1e-10

    def test_fit_reference_raises_on_too_few_observations(self):
        prod, comp = _build_dgp(start_year=1947, end_year=1947)
        with pytest.raises(ValueError, match="reference window"):
            _fit_reference(prod, comp, 1900, 1900)

    def test_build_cf_series_starts_at_pivot_actual(self):
        prod, comp = _build_dgp()
        post_idx = comp.loc["1972-01-01":].index
        cf = _build_cf_series(prod, comp, alpha=0.0, beta=1.0, post_idx=post_idx)
        assert cf.iloc[0] == pytest.approx(comp.loc[post_idx[0]])
        assert cf.name == "counterfactual_compensation"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_panel_and_audit(self, tmp_path: Path):
        prod, comp = _build_dgp(pre_beta=1.0, post_beta=0.5)
        result = compute_counterfactual(prod, comp, n_bootstrap=50)
        paths = save_counterfactual_result(result, tmp_path)
        assert paths["panel"].exists()
        assert paths["audit"].exists()

    def test_panel_has_four_columns(self, tmp_path: Path):
        prod, comp = _build_dgp(pre_beta=1.0, post_beta=0.5)
        result = compute_counterfactual(prod, comp, n_bootstrap=50)
        paths = save_counterfactual_result(result, tmp_path)
        df = pd.read_csv(paths["panel"], index_col=0, parse_dates=True)
        for col in (
            "actual_compensation", "counterfactual_compensation",
            "gap", "pct_gap",
        ):
            assert col in df.columns

    def test_audit_json_parseable(self, tmp_path: Path):
        prod, comp = _build_dgp(pre_beta=1.0, post_beta=0.5)
        result = compute_counterfactual(prod, comp, n_bootstrap=50)
        paths = save_counterfactual_result(result, tmp_path)
        with open(paths["audit"]) as fh:
            d = json.load(fh)
        assert "final_pct_gap" in d
        assert "final_pct_gap_ci_low" in d
        assert "final_pct_gap_ci_high" in d

    def test_custom_prefix(self, tmp_path: Path):
        prod, comp = _build_dgp()
        result = compute_counterfactual(prod, comp, n_bootstrap=20)
        paths = save_counterfactual_result(result, tmp_path, prefix="lost_welfare")
        assert paths["panel"].name == "lost_welfare_panel.csv"
        assert paths["audit"].name == "lost_welfare_audit.json"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_empty_productivity_raises(self):
        idx = _quarterly_index(1947, 2024)
        comp = pd.Series(np.ones(len(idx)), index=idx)
        with pytest.raises(ValueError, match="non-empty"):
            compute_counterfactual(pd.Series(dtype=float), comp, n_bootstrap=10)

    def test_empty_compensation_raises(self):
        idx = _quarterly_index(1947, 2024)
        prod = pd.Series(np.ones(len(idx)), index=idx)
        with pytest.raises(ValueError, match="non-empty"):
            compute_counterfactual(prod, pd.Series(dtype=float), n_bootstrap=10)

    def test_no_overlap_raises(self):
        idx_a = _quarterly_index(1900, 1910)
        idx_b = _quarterly_index(2000, 2010)
        prod = pd.Series(np.ones(len(idx_a)), index=idx_a)
        comp = pd.Series(np.ones(len(idx_b)), index=idx_b)
        with pytest.raises(ValueError, match="insufficient overlap"):
            compute_counterfactual(prod, comp, n_bootstrap=10)

    def test_reference_end_after_data_raises(self):
        # Reference window ends in 2050, but data only runs to 2024.
        prod, comp = _build_dgp(start_year=1947, end_year=2024)
        with pytest.raises(ValueError, match="reference window"):
            compute_counterfactual(
                prod, comp,
                reference_start=2030,
                reference_end=2050,
                n_bootstrap=10,
            )

    def test_reference_window_too_narrow_raises(self):
        prod, comp = _build_dgp(start_year=1947, end_year=2024)
        # A single-year window — only 4 quarterly observations, fewer than 5
        # after differencing.
        with pytest.raises(ValueError, match="reference window"):
            compute_counterfactual(
                prod, comp,
                reference_start=1960,
                reference_end=1960,
                n_bootstrap=10,
            )

    def test_post_reference_too_narrow_raises(self):
        # Force the reference window to extend almost to the end of the data.
        prod, comp = _build_dgp(start_year=1947, end_year=2024)
        with pytest.raises(ValueError, match="post-reference"):
            compute_counterfactual(
                prod, comp,
                reference_start=1948,
                reference_end=2024,
                n_bootstrap=10,
            )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_identical_inputs_yield_identical_results(self):
        prod, comp = _build_dgp(
            pre_beta=1.0, post_beta=0.5, noise_sd=0.005, seed=99,
        )
        a = compute_counterfactual(prod, comp, n_bootstrap=100, random_seed=42)
        b = compute_counterfactual(prod, comp, n_bootstrap=100, random_seed=42)
        pd.testing.assert_series_equal(a.counterfactual, b.counterfactual)
        pd.testing.assert_series_equal(a.gap, b.gap)
        assert a.final_pct_gap == b.final_pct_gap
        assert a.final_pct_gap_ci == b.final_pct_gap_ci
        assert a.reference_alpha == b.reference_alpha
        assert a.reference_beta == b.reference_beta


# ---------------------------------------------------------------------------
# Substantive scenario: H4 plausibility check
# ---------------------------------------------------------------------------

class TestH4PlausibilityScenario:
    """A sanity check that the registered H4 threshold (≥20% cumulative gap
    by 2025) is *detectable* by the procedure under a plausible synthetic DGP.

    Generates a 1947-2024 series in which workers capture full productivity
    growth pre-1971 (β=1.0) and only 60% post-1971 (β=0.6) — close to the
    Mishel-Bivens (2015) baseline. Verifies that the procedure flags a gap
    well in excess of 20%.
    """

    def test_plausible_dgp_produces_substantial_gap(self):
        prod, comp = _build_dgp(
            start_year=1947, end_year=2024,
            pre_alpha=0.0, pre_beta=1.0,
            post_alpha=0.0, post_beta=0.6,
            prod_q_growth=0.005,
            noise_sd=0.0,
            pivot_year=1971,
        )
        result = compute_counterfactual(
            prod, comp,
            reference_start=1948,
            reference_end=1971,
            n_bootstrap=200,
        )
        # 53 years × ~0.4pp annualized decoupling × compounding → 20%+ gap.
        assert result.final_pct_gap > 0.20
        # Bootstrap CI should exclude zero (no-noise DGP, so this is tight).
        assert result.final_pct_gap_ci[0] > 0.0
