"""
rpps.counterfactual — counterfactual real-compensation trajectory.

Implements §5.4 of:
    Green, R. J. (2026). The Inflationary Yardstick. Working Paper.

Tests hypothesis H4 (counterfactual gap): the counterfactual exercise applying
1948–1971 productivity-distribution coefficients to the realized 1972–2025
productivity path produces a cumulative real-compensation gap by 2025 of at
least 20% of actual real compensation, with bootstrap confidence intervals
excluding zero.

Method
------
Step 1. Estimate the productivity-distribution relationship in the reference
window (default 1948–1971):

    Δlog(C_t) = α_0 + β_0 · Δlog(Q_t) + ε_t,    t in [t_start, t_end]

via OLS. Save (α̂_0, β̂_0, residuals, residual variance).

Step 2. Construct the counterfactual real-compensation series for the
post-reference period using the actually realized productivity path:

    Ĉ^cf_t = Ĉ^cf_{t-1} · exp(α̂_0 + β̂_0 · Δlog(Q_t))

with Ĉ^cf at the reference-window end pinned to the actual real-compensation
level at that date.

Step 3. Compute the gap series:

    gap_t = Ĉ^cf_t - C^actual_t

and the final-period percentage gap:

    pct_gap_T = (Ĉ^cf_T - C^actual_T) / C^actual_T

Step 4. Bootstrap confidence intervals. Resample residuals from Step 1 with
replacement, refit the counterfactual with each bootstrap sample's coefficients,
and report the empirical 2.5/97.5 percentiles of pct_gap_T as the 95% CI.

The counterfactual is not a forecast of what would have occurred; it is a
benchmark quantifying the cumulative welfare effect of the post-1971 regime
change relative to a counterfactual continuation of the prior regime. The
difference does not, by itself, demonstrate that the prior regime was sustainable
in perpetuity, nor that policy choices were proximate causes of the regime
change. It measures the gap; it does not argue about its causes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_REFERENCE_START = 1948
DEFAULT_REFERENCE_END = 1971
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualResult:
    """Result of the counterfactual real-compensation exercise.

    Attributes
    ----------
    counterfactual : pd.Series
        Counterfactual real-compensation series, post-reference-window.
    actual : pd.Series
        Actual real-compensation series, post-reference-window.
    gap : pd.Series
        counterfactual − actual, in the same units as compensation.
    pct_gap : pd.Series
        (counterfactual − actual) / actual, expressed as a fraction.
    final_pct_gap : float
        pct_gap at the last observation.
    final_pct_gap_ci : tuple[float, float]
        Bootstrap 95% confidence interval for final_pct_gap.
    reference_alpha : float
    reference_beta : float
    reference_n : int
    n_bootstrap : int
    audit : dict
    """

    counterfactual: pd.Series
    actual: pd.Series
    gap: pd.Series
    pct_gap: pd.Series
    final_pct_gap: float
    final_pct_gap_ci: tuple[float, float]
    reference_alpha: float
    reference_beta: float
    reference_n: int
    n_bootstrap: int
    audit: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "reference_alpha": self.reference_alpha,
            "reference_beta": self.reference_beta,
            "reference_n": self.reference_n,
            "n_bootstrap": self.n_bootstrap,
            "final_pct_gap": self.final_pct_gap,
            "final_pct_gap_ci_low": self.final_pct_gap_ci[0],
            "final_pct_gap_ci_high": self.final_pct_gap_ci[1],
            "audit": self.audit,
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_counterfactual(
    productivity: pd.Series,
    real_compensation: pd.Series,
    *,
    reference_start: int = DEFAULT_REFERENCE_START,
    reference_end: int = DEFAULT_REFERENCE_END,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = DEFAULT_RANDOM_SEED,
    confidence_level: float = 0.95,
) -> CounterfactualResult:
    """Compute the counterfactual real-compensation trajectory.

    Parameters
    ----------
    productivity : pd.Series
        Productivity index or level (e.g. OPHNFB).
    real_compensation : pd.Series
        Real-compensation index or level (e.g. COMPRNFB).
    reference_start : int, default 1948
        Start year of the reference window from which distribution coefficients
        are estimated.
    reference_end : int, default 1971
        End year of the reference window (inclusive).
    n_bootstrap : int, default 1000
        Number of bootstrap iterations for the confidence interval.
    random_seed : int, default 42
        Seed for deterministic bootstrap.
    confidence_level : float, default 0.95
        Confidence level for the bootstrap interval.

    Returns
    -------
    CounterfactualResult
    """
    if productivity.empty or real_compensation.empty:
        raise ValueError("inputs must be non-empty")

    prod = _to_dt_idx(productivity).sort_index()
    comp = _to_dt_idx(real_compensation).sort_index()

    common = prod.index.intersection(comp.index)
    if len(common) < 10:
        raise ValueError("insufficient overlap between productivity and compensation")
    prod = prod.loc[common]
    comp = comp.loc[common]

    # Step 1. Estimate reference-window coefficients.
    alpha_hat, beta_hat, residuals, ref_n = _fit_reference(
        prod, comp, reference_start, reference_end
    )

    # Step 2. Build counterfactual on the post-reference path.
    pivot_date = _last_date_in_year(comp, reference_end)
    if pivot_date is None:
        raise ValueError(f"reference_end={reference_end} not found in compensation index")

    post_idx = comp.index[comp.index >= pivot_date]
    if len(post_idx) < 2:
        raise ValueError("post-reference window has fewer than 2 observations")

    cf_series = _build_cf_series(prod, comp, alpha_hat, beta_hat, post_idx)
    actual_series = comp.loc[post_idx]

    gap = cf_series - actual_series
    pct_gap = gap / actual_series

    final_pct = float(pct_gap.iloc[-1])

    # Step 4. Bootstrap.
    ci = _bootstrap_final_pct_gap(
        prod=prod,
        comp=comp,
        residuals=residuals,
        post_idx=post_idx,
        reference_start=reference_start,
        reference_end=reference_end,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        confidence_level=confidence_level,
    )

    audit = {
        "reference_window_years": [reference_start, reference_end],
        "reference_alpha": alpha_hat,
        "reference_beta": beta_hat,
        "reference_n_observations": ref_n,
        "post_reference_n": len(post_idx),
        "post_reference_start": str(pivot_date),
        "post_reference_end": str(post_idx[-1]),
        "n_bootstrap": n_bootstrap,
        "random_seed": random_seed,
        "confidence_level": confidence_level,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return CounterfactualResult(
        counterfactual=cf_series,
        actual=actual_series,
        gap=gap,
        pct_gap=pct_gap,
        final_pct_gap=final_pct,
        final_pct_gap_ci=ci,
        reference_alpha=float(alpha_hat),
        reference_beta=float(beta_hat),
        reference_n=ref_n,
        n_bootstrap=n_bootstrap,
        audit=audit,
    )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _to_dt_idx(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        s = s.copy()
        s.index = pd.DatetimeIndex(s.index)
    return s


def _last_date_in_year(s: pd.Series, year: int) -> pd.Timestamp | None:
    """Return the latest observation date in the given calendar year."""
    in_year = s[s.index.year == year]
    if in_year.empty:
        return None
    return in_year.index[-1]


def _fit_reference(
    prod: pd.Series,
    comp: pd.Series,
    start_year: int,
    end_year: int,
) -> tuple[float, float, np.ndarray, int]:
    """OLS fit Δlog(C) = α + β·Δlog(Q) + ε on the reference window."""
    mask = (comp.index.year >= start_year) & (comp.index.year <= end_year)
    if mask.sum() < 5:
        raise ValueError(
            f"reference window {start_year}-{end_year} has only {mask.sum()} observations"
        )

    log_comp = np.log(comp.loc[mask].astype(float))
    log_prod = np.log(prod.loc[mask].astype(float))
    d_log_comp = log_comp.diff().dropna()
    d_log_prod = log_prod.diff().dropna()

    joint = d_log_comp.index.intersection(d_log_prod.index)
    y = d_log_comp.loc[joint].to_numpy()
    x = d_log_prod.loc[joint].to_numpy()

    n = len(y)
    if n < 5:
        raise ValueError(f"insufficient differenced observations: {n}")

    X = np.column_stack([np.ones(n), x])
    beta_full, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha_hat = float(beta_full[0])
    beta_hat = float(beta_full[1])
    residuals = y - X @ beta_full
    return alpha_hat, beta_hat, residuals, n


def _build_cf_series(
    prod: pd.Series,
    comp: pd.Series,
    alpha: float,
    beta: float,
    post_idx: pd.DatetimeIndex,
) -> pd.Series:
    """Construct the counterfactual real-compensation series."""
    log_prod = np.log(prod.astype(float))
    d_log_prod_post = log_prod.loc[post_idx].diff()

    pivot_actual = float(comp.loc[post_idx[0]])
    cf_values = np.empty(len(post_idx))
    cf_values[0] = pivot_actual

    for i in range(1, len(post_idx)):
        delta = d_log_prod_post.iloc[i]
        if not np.isfinite(delta):
            cf_values[i] = cf_values[i - 1]
        else:
            cf_values[i] = cf_values[i - 1] * np.exp(alpha + beta * delta)

    return pd.Series(cf_values, index=post_idx, name="counterfactual_compensation")


def _bootstrap_final_pct_gap(
    prod: pd.Series,
    comp: pd.Series,
    residuals: np.ndarray,
    post_idx: pd.DatetimeIndex,
    reference_start: int,
    reference_end: int,
    n_bootstrap: int,
    random_seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Residual-bootstrap the final-period percentage gap."""
    rng = np.random.default_rng(random_seed)

    # Reference-window inputs for bootstrap re-fitting.
    mask = (comp.index.year >= reference_start) & (comp.index.year <= reference_end)
    log_comp = np.log(comp.loc[mask].astype(float))
    log_prod = np.log(prod.loc[mask].astype(float))
    d_log_comp = log_comp.diff().dropna()
    d_log_prod = log_prod.diff().dropna()
    joint = d_log_comp.index.intersection(d_log_prod.index)
    x_ref = d_log_prod.loc[joint].to_numpy()
    y_ref_fitted = d_log_comp.loc[joint].to_numpy() - residuals
    n_ref = len(residuals)

    final_gaps = np.empty(n_bootstrap)
    actual_final = float(comp.loc[post_idx[-1]])

    for b in range(n_bootstrap):
        boot_resid = rng.choice(residuals, size=n_ref, replace=True)
        y_boot = y_ref_fitted + boot_resid
        X_ref = np.column_stack([np.ones(n_ref), x_ref])
        beta_b, _, _, _ = np.linalg.lstsq(X_ref, y_boot, rcond=None)
        cf_b = _build_cf_series(prod, comp, float(beta_b[0]), float(beta_b[1]), post_idx)
        gap_b = float(cf_b.iloc[-1] - actual_final) / actual_final
        final_gaps[b] = gap_b

    alpha = (1.0 - confidence_level) / 2.0
    lo = float(np.quantile(final_gaps, alpha))
    hi = float(np.quantile(final_gaps, 1.0 - alpha))
    return lo, hi


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_counterfactual_result(
    result: CounterfactualResult,
    output_dir: str | Path,
    prefix: str = "counterfactual",
) -> dict[str, Path]:
    """Save CounterfactualResult to CSV + JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    panel_path = out / f"{prefix}_panel.csv"
    audit_path = out / f"{prefix}_audit.json"

    panel = pd.concat(
        [
            result.actual.rename("actual_compensation"),
            result.counterfactual,
            result.gap.rename("gap"),
            result.pct_gap.rename("pct_gap"),
        ],
        axis=1,
    )
    panel.to_csv(panel_path)
    with open(audit_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)

    return {"panel": panel_path, "audit": audit_path}
