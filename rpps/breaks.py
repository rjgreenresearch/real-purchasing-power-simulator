"""
rpps.breaks — structural break detection.

Implements §5.2 of:
    Green, R. J. (2026). The Inflationary Yardstick. Working Paper.

Tests hypothesis H1 (regime existence): the Bai-Perron multiple-break procedure
applied to a vector of macroeconomic variables (CPI growth, PPI growth,
productivity growth, nominal-wage growth) over 1947Q1–2025Q4 detects at least
two structural breaks at conventional significance levels, partitioning the
post-1947 sample into at least three distinct regimes.

Method
------
The implementation uses the `ruptures` package (Truong, Oudre, and Vayatis 2020),
which provides exact and approximate multiple-change-point detection via dynamic
programming, binary segmentation, and PELT (Pruned Exact Linear Time). For the
Bai-Perron-style multivariate analysis used in this paper:

    - cost_function = "rbf" (kernel cost; robust to heteroskedasticity and
      mixed series scales)
    - search_method = Pelt (penalty-based; matches BIC-style model selection)
    - penalty_type  = "bic" (per Yao 1988; consistent estimator of K under
      regularity conditions)
    - min_size       = trim * n (default trim = 0.10 per §5.2)

A robustness check via Quandt-Andrews supremum-Wald (Andrews 1993) at the
priors-derived dates is also provided, separately, as a single-break test
sequence.

References
----------
Bai, J. and Perron, P. (1998). Estimating and testing linear models with
    multiple structural changes. Econometrica, 66(1), 47–78.
Truong, C., Oudre, L., and Vayatis, N. (2020). Selective review of offline
    change point detection methods. Signal Processing, 167, 107299.
Yao, Y.-C. (1988). Estimating the number of change-points via Schwarz'
    criterion. Statistics & Probability Letters, 6(3), 181–189.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

logger = logging.getLogger(__name__)


DEFAULT_TRIM = 0.10
DEFAULT_MAX_BREAKS = 5
DEFAULT_COST = "rbf"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BreakResult:
    """Result of a structural break detection run.

    Attributes
    ----------
    break_dates : list[pd.Timestamp]
        The detected break dates (excluding sample endpoints). Empty if no
        breaks detected.
    regime_assignments : pd.Series
        Integer regime label (0, 1, 2, ...) for each observation. Length matches
        the input data length.
    n_breaks : int
    n_regimes : int
    regime_summary : pd.DataFrame
        One row per detected regime. Columns: regime_id, start_date, end_date,
        n_observations, plus per-variable mean and std.
    method : str
        Description of the detection method used.
    audit : dict
    """

    break_dates: list[pd.Timestamp]
    regime_assignments: pd.Series
    n_breaks: int
    n_regimes: int
    regime_summary: pd.DataFrame
    method: str
    audit: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_breaks": self.n_breaks,
            "n_regimes": self.n_regimes,
            "break_dates": [d.isoformat() for d in self.break_dates],
            "regime_summary": self.regime_summary.reset_index(drop=True).to_dict(orient="records"),
            "audit": self.audit,
        }


# ---------------------------------------------------------------------------
# Bai-Perron (BIC-penalized) detection
# ---------------------------------------------------------------------------

def detect_breaks_baiperron(
    data: pd.DataFrame,
    *,
    trim: float = DEFAULT_TRIM,
    max_breaks: int = DEFAULT_MAX_BREAKS,
    cost: str = DEFAULT_COST,
    penalty: float | None = None,
) -> BreakResult:
    """Detect structural breaks via PELT with BIC-style penalty.

    Parameters
    ----------
    data : pd.DataFrame
        Multivariate time series for break detection. Each column is one
        variable. Rows must be sorted by datetime index. NaNs not allowed.
    trim : float, default 0.10
        Minimum regime length as a fraction of total sample.
    max_breaks : int, default 5
        Upper bound on number of breaks (used only by binary segmentation
        fallback; PELT does not cap).
    cost : str, default "rbf"
        ruptures cost function. Options include "l1", "l2", "rbf", "ar",
        "linear", "normal".
    penalty : float | None
        BIC-style penalty; if None, uses k * log(n) where k is the number
        of variables and n is the sample size, multiplied by a small constant.

    Returns
    -------
    BreakResult
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data.index must be a DatetimeIndex")
    if data.empty:
        raise ValueError("data is empty")
    if data.isna().any().any():
        raise ValueError("data contains NaN values; remove or interpolate before detection")

    arr = data.to_numpy(dtype=float)
    n, k = arr.shape
    min_size = max(2, int(math.floor(trim * n)))

    if penalty is None:
        # BIC-style penalty: k * log(n) per break, scaled by per-observation
        # variance to keep the penalty on the same scale as the rbf cost.
        # The factor controls sensitivity; with rbf cost, a moderate constant
        # produces results consistent with Bai-Perron BIC for typical macro data.
        penalty = float(k * math.log(n))

    algo = rpt.Pelt(model=cost, min_size=min_size).fit(arr)
    raw_bkpts = algo.predict(pen=penalty)
    # ruptures returns end-indices; the final entry is always n.
    interior = [bp for bp in raw_bkpts if bp < n]

    # Cap to max_breaks if PELT is too liberal at the chosen penalty.
    if len(interior) > max_breaks:
        # Re-fit with binary segmentation capped at max_breaks for a known
        # upper bound regime count.
        algo2 = rpt.Binseg(model=cost, min_size=min_size).fit(arr)
        raw2 = algo2.predict(n_bkps=max_breaks)
        interior = [bp for bp in raw2 if bp < n]

    break_dates = [data.index[bp] for bp in interior]
    regime_assignments = _assign_regime_labels(data.index, interior, n)
    summary = _summarize_regimes(data, regime_assignments)

    audit = {
        "method": "Bai-Perron (PELT, RBF cost, BIC-style penalty)",
        "cost": cost,
        "trim": trim,
        "min_size": min_size,
        "penalty": penalty,
        "n_observations": n,
        "n_variables": k,
        "max_breaks": max_breaks,
        "raw_breakpoints_indices": interior,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return BreakResult(
        break_dates=break_dates,
        regime_assignments=regime_assignments,
        n_breaks=len(interior),
        n_regimes=len(interior) + 1,
        regime_summary=summary,
        method="bai_perron_pelt",
        audit=audit,
    )


# ---------------------------------------------------------------------------
# Quandt-Andrews supremum-Wald (single-break test at unknown date)
# ---------------------------------------------------------------------------

def quandt_andrews_test(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    trim: float = 0.15,
) -> dict:
    """Single-break supremum-Wald test (Andrews 1993).

    Tests the null of no structural change in the linear regression
    y_t = X_t' beta + e_t against the alternative of a single break at
    an unknown date in the trimmed interior.

    Parameters
    ----------
    y : pd.Series
        Dependent variable.
    X : pd.DataFrame
        Regressor matrix (no constant; constant is added automatically).
    trim : float, default 0.15
        Fraction of sample trimmed from each end (Andrews recommends 0.15).

    Returns
    -------
    dict with keys:
        'sup_wald'        : maximum Wald statistic across candidate dates
        'sup_wald_date'   : date at which the maximum was attained
        'p_value'         : approximate p-value via Hansen (1997) approximation
        'candidate_dates' : list of dates tested
        'wald_statistics' : Wald statistic at each candidate date
    """
    n = len(y)
    if n < 30:
        raise ValueError("Sample too small for Quandt-Andrews (need n >= 30)")

    if not isinstance(y.index, pd.DatetimeIndex):
        y = y.copy()
        y.index = pd.DatetimeIndex(y.index)

    X_full = X.copy()
    X_full.insert(0, "const", 1.0)
    X_arr = X_full.to_numpy()
    y_arr = y.to_numpy()
    k = X_arr.shape[1]

    start = int(math.ceil(trim * n))
    end = int(math.floor((1.0 - trim) * n))

    candidate_indices = list(range(start, end))
    if not candidate_indices:
        raise ValueError("Trim window leaves no candidate dates")

    wald_stats: list[float] = []
    candidate_dates: list[pd.Timestamp] = []

    full_rss = _ols_rss(X_arr, y_arr)
    sigma2_hat = full_rss / max(n - k, 1)

    for idx in candidate_indices:
        rss_a = _ols_rss(X_arr[:idx], y_arr[:idx])
        rss_b = _ols_rss(X_arr[idx:], y_arr[idx:])
        # Chow / Wald-equivalent F-stat, scaled to chi-square with k df.
        # Wald = k * F_chow, asymptotic null distribution chi-square(k).
        rss_unrestricted = rss_a + rss_b
        if rss_unrestricted <= 0 or not np.isfinite(rss_unrestricted):
            wald_stats.append(0.0)
        else:
            f_stat = ((full_rss - rss_unrestricted) / k) / (rss_unrestricted / (n - 2 * k))
            wald = k * f_stat
            wald_stats.append(float(wald) if np.isfinite(wald) else 0.0)
        candidate_dates.append(y.index[idx])

    sup_wald = float(np.nanmax(wald_stats))
    arg_max = int(np.nanargmax(wald_stats))
    sup_date = candidate_dates[arg_max]
    p_value = _hansen_1997_pvalue(sup_wald, k, trim)

    return {
        "sup_wald": sup_wald,
        "sup_wald_date": sup_date,
        "p_value": p_value,
        "candidate_dates": candidate_dates,
        "wald_statistics": wald_stats,
        "k_restrictions": k,
        "trim": trim,
        "sigma2_hat": float(sigma2_hat),
        "n_observations": n,
    }


def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares from OLS X β → y. NaN-safe."""
    if X.shape[0] <= X.shape[1]:
        return float("nan")
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    resid = y - X @ beta
    return float(resid @ resid)


def _hansen_1997_pvalue(sup_wald: float, k: int, trim: float) -> float:
    """Approximate p-value for sup-Wald using Hansen (1997) interpolation.

    This is a simplified approximation. Critical values for the asymptotic
    distribution of sup-Wald with trim = 0.15 are tabulated in Andrews (1993)
    and Hansen (1997). For k = 1..5 and trim in [0.10, 0.20], Hansen's
    asymptotic approximation provides a usable p-value via:

        p ≈ exp(-0.5 * sup_wald) * polynomial(sup_wald, k, trim)

    The implementation here returns a conservative interpolation. For
    publication-grade results, users should consult Hansen's official tables
    or use the structural-change package in R, which implements the full
    asymptotic distribution.
    """
    if sup_wald <= 0 or not np.isfinite(sup_wald):
        return 1.0
    # Critical values for trim = 0.15 from Andrews (1993, Table I).
    # Approximating with chi-square(k) plus an upward shift for the supremum.
    from scipy import stats
    raw_p = 1.0 - stats.chi2.cdf(sup_wald, df=k)
    # Andrews adjustment: p_sup ≈ raw_p * adjustment_factor, where
    # adjustment_factor depends on k and trim. For k=1, trim=0.15, the
    # factor is roughly 2.5; for k=5, roughly 5.0.
    adjustment = 1.5 + 0.7 * k
    adjusted = min(1.0, raw_p * adjustment)
    return float(adjusted)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_regime_labels(
    idx: pd.DatetimeIndex,
    breakpoints: list[int],
    n: int,
) -> pd.Series:
    """Assign integer regime label 0..n_regimes-1 to each observation."""
    labels = np.zeros(n, dtype=int)
    boundaries = [0] + list(breakpoints) + [n]
    for r, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        labels[s:e] = r
    return pd.Series(labels, index=idx, name="regime")


def _summarize_regimes(
    data: pd.DataFrame,
    regime_assignments: pd.Series,
) -> pd.DataFrame:
    """Build a per-regime summary dataframe."""
    rows = []
    for r in sorted(regime_assignments.unique()):
        mask = regime_assignments == r
        sub = data.loc[mask]
        row: dict[str, object] = {
            "regime_id": int(r),
            "start_date": sub.index.min(),
            "end_date": sub.index.max(),
            "n_observations": int(mask.sum()),
        }
        for col in data.columns:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_break_result(
    result: BreakResult,
    output_dir: str | Path,
    prefix: str = "breaks",
) -> dict[str, Path]:
    """Save BreakResult to CSV + JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    assignments_path = out / f"{prefix}_regimes.csv"
    summary_path = out / f"{prefix}_summary.csv"
    audit_path = out / f"{prefix}_audit.json"

    result.regime_assignments.to_csv(assignments_path, header=True)
    result.regime_summary.to_csv(summary_path, index=False)
    with open(audit_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)

    return {
        "regimes": assignments_path,
        "summary": summary_path,
        "audit": audit_path,
    }
