"""
rpps.metrics.wicr — Wage-Inflation Capture Ratio.

Implements §5.1, metric #2 of:
    Green, R. J. (2026). The Inflationary Yardstick. Working Paper.

Definition
----------
For nominal wage w_t with year-over-year growth rate g^w_t and CPI with
year-over-year growth rate π_t:

    WICR_t = π_t / g^w_t

Interpretation
--------------
    WICR < 0     :  one of inflation/wages is negative; sign analysis required
    WICR = 0     :  no inflation; full nominal-wage gain converts to real gain
    0 < WICR < 1 :  positive real-wage growth; share (1 - WICR) is real gain
    WICR ≈ 1     :  real wages flat; nominal gain entirely absorbed by inflation
    WICR > 1     :  real wages declining; inflation outpaced wage growth

Threshold hypothesis (H3, §3.3, §6.1)
-------------------------------------
The pre-registered threshold hypothesis is that when WICR exceeds approximately
0.80 sustainedly (4-quarter moving average above 0.80 for at least 8 consecutive
quarters, per §6.1), the wage-welfare elasticity falls substantially. This
module computes WICR and the threshold-regime indicator; the elasticity-by-regime
test is in `rpps.regression`.

Usage
-----
    from rpps.metrics.wicr import compute_wicr
    result = compute_wicr(wage_series, cpi_series)
    result.wicr_yoy        # raw YoY ratio
    result.wicr_smoothed   # 4-quarter moving average (or 12-month for monthly)
    result.regime_label    # categorical: low / medium / high (relative to thresholds)
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


# Threshold parameters (registered priors; falsifiable in Stage 2).
WICR_LOW_THRESHOLD = 0.50
WICR_HIGH_THRESHOLD = 0.80
SUSTAINED_QUARTERS = 8  # H3 sustaining condition


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WicrResult:
    """Container for Wage-Inflation Capture Ratio computation.

    Attributes
    ----------
    wicr_yoy : pd.Series
        Raw period-by-period WICR = π_t / g_w_t. May contain extreme values
        when g_w_t is near zero.
    wicr_smoothed : pd.Series
        Moving-average smoothed WICR. Default smoothing window is 12 months
        for monthly data and 4 quarters for quarterly data.
    wage_growth_yoy : pd.Series
        Year-over-year nominal wage growth rate (decimal, not percent).
    inflation_yoy : pd.Series
        Year-over-year CPI growth rate (decimal, not percent).
    regime_label : pd.Series
        Categorical: 'low' (WICR < 0.50), 'medium' (0.50-0.80), 'high' (>0.80).
        Computed on smoothed WICR.
    high_wicr_runs : pd.Series
        Boolean series: True when smoothed WICR has been above 0.80 for at
        least SUSTAINED_QUARTERS consecutive periods. Used by the H3 test
        in `rpps.regression`.
    n_observations : int
    audit : dict
    """

    wicr_yoy: pd.Series
    wicr_smoothed: pd.Series
    wage_growth_yoy: pd.Series
    inflation_yoy: pd.Series
    regime_label: pd.Series
    high_wicr_runs: pd.Series
    n_observations: int
    audit: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_observations": self.n_observations,
            "n_high_wicr_periods": int(self.high_wicr_runs.sum()),
            "regime_distribution": self.regime_label.value_counts(dropna=True).to_dict(),
            "audit": self.audit,
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_wicr(
    wage_series: pd.Series,
    cpi_series: pd.Series,
    *,
    smoothing_periods: int | None = None,
    low_threshold: float = WICR_LOW_THRESHOLD,
    high_threshold: float = WICR_HIGH_THRESHOLD,
    sustained_periods: int = SUSTAINED_QUARTERS,
) -> WicrResult:
    """Compute WICR from wage and CPI level series.

    Parameters
    ----------
    wage_series : pd.Series
        Nominal hourly wage (level), e.g. AHETPI. Must be monthly or quarterly.
    cpi_series : pd.Series
        Consumer Price Index (level), e.g. CPIAUCNS. Must be monthly or quarterly.
    smoothing_periods : int | None
        Periods for the moving-average smoothing window. If None, inferred from
        the dominant series frequency: 12 for monthly, 4 for quarterly.
    low_threshold : float, default 0.50
        WICR cutoff between 'low' and 'medium' regime labels.
    high_threshold : float, default 0.80
        WICR cutoff between 'medium' and 'high' regime labels (the H3 prior).
    sustained_periods : int, default 8
        Number of consecutive periods above high_threshold required to mark
        the period as a sustained-high-WICR run (the H3 test condition).

    Returns
    -------
    WicrResult

    Raises
    ------
    ValueError
        If either input is empty, or if the joint coverage is insufficient
        (less than 13 observations needed for YoY growth).
    """
    if wage_series.empty:
        raise ValueError("wage_series is empty")
    if cpi_series.empty:
        raise ValueError("cpi_series is empty")

    wage = _to_datetime_index(wage_series).sort_index()
    cpi = _to_datetime_index(cpi_series).sort_index()

    # Align to common frequency. Use the more granular of the two.
    wage_freq = _detect_freq(wage.index)
    cpi_freq = _detect_freq(cpi.index)
    target_freq = _pick_target_freq(wage_freq, cpi_freq)

    wage_r = _resample(wage, target_freq)
    cpi_r = _resample(cpi, target_freq)

    common_idx = wage_r.index.intersection(cpi_r.index)
    if len(common_idx) < 13:
        raise ValueError(
            f"Insufficient overlap: {len(common_idx)} periods. Need at least 13 for YoY."
        )
    wage_aligned = wage_r.loc[common_idx]
    cpi_aligned = cpi_r.loc[common_idx]

    # YoY growth rates (period-over-period at 12-month or 4-quarter lag).
    yoy_lag = 12 if target_freq == "M" else 4
    wage_growth = wage_aligned.pct_change(periods=yoy_lag)
    inflation = cpi_aligned.pct_change(periods=yoy_lag)
    wage_growth.name = "wage_growth_yoy"
    inflation.name = "inflation_yoy"

    # Raw WICR. Avoid division-by-near-zero artifacts via masking.
    eps = 1e-6
    safe_wage_growth = wage_growth.where(wage_growth.abs() > eps, np.nan)
    wicr = inflation / safe_wage_growth
    wicr.name = "wicr_yoy"

    # Smoothed WICR (centered=False, trailing window for online interpretability).
    if smoothing_periods is None:
        smoothing_periods = yoy_lag
    wicr_smoothed = wicr.rolling(window=smoothing_periods, min_periods=smoothing_periods // 2).mean()
    wicr_smoothed.name = "wicr_smoothed"

    # Regime labels on smoothed WICR.
    regime = pd.Series(index=wicr_smoothed.index, dtype="object", name="regime_label")
    regime[wicr_smoothed < low_threshold] = "low"
    regime[(wicr_smoothed >= low_threshold) & (wicr_smoothed <= high_threshold)] = "medium"
    regime[wicr_smoothed > high_threshold] = "high"

    # Sustained-high runs: True at periods within an unbroken run of
    # smoothed WICR > high_threshold of length >= sustained_periods.
    above = (wicr_smoothed > high_threshold).fillna(False)
    high_wicr_runs = _flag_sustained_runs(above, sustained_periods)
    high_wicr_runs.name = "high_wicr_run"

    audit = {
        "computation": "WICR = inflation_yoy / wage_growth_yoy (smoothed)",
        "yoy_lag_periods": yoy_lag,
        "target_frequency": target_freq,
        "smoothing_periods": smoothing_periods,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "sustained_periods": sustained_periods,
        "wage_input_n_obs": len(wage_series),
        "cpi_input_n_obs": len(cpi_series),
        "joint_n_obs": len(common_idx),
        "n_high_wicr_run_periods": int(high_wicr_runs.sum()),
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return WicrResult(
        wicr_yoy=wicr,
        wicr_smoothed=wicr_smoothed,
        wage_growth_yoy=wage_growth,
        inflation_yoy=inflation,
        regime_label=regime,
        high_wicr_runs=high_wicr_runs,
        n_observations=int(wicr.notna().sum()),
        audit=audit,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_datetime_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        s = s.copy()
        s.index = pd.DatetimeIndex(s.index)
    return s


def _detect_freq(idx: pd.DatetimeIndex) -> str:
    """Return 'M' for monthly-like, 'Q' for quarterly-like, 'A' for annual-like."""
    # pd.infer_freq raises ValueError on indices with fewer than 3 dates,
    # so handle small indices before attempting inference.
    if len(idx) < 3:
        if len(idx) < 2:
            return "M"
        diffs = pd.Series(idx).diff().dropna()
        median_days = diffs.dt.days.median()
        if median_days <= 35:
            return "M"
        if median_days <= 100:
            return "Q"
        return "A"

    inferred = pd.infer_freq(idx) or ""
    inferred_upper = inferred.upper() if inferred else ""
    if inferred_upper.startswith(("M", "MS")) or "M" in inferred_upper[:2]:
        return "M"
    if inferred_upper.startswith(("Q", "QS")):
        return "Q"
    if inferred_upper.startswith(("A", "Y", "AS", "YS")):
        return "A"
    # Fallback: use median spacing.
    diffs = pd.Series(idx).diff().dropna()
    median_days = diffs.dt.days.median()
    if median_days <= 35:
        return "M"
    if median_days <= 100:
        return "Q"
    return "A"


def _pick_target_freq(a: str, b: str) -> str:
    """Pick the more granular of two frequencies."""
    order = {"M": 0, "Q": 1, "A": 2}
    return a if order.get(a, 2) <= order.get(b, 2) else b


def _resample(s: pd.Series, target_freq: str) -> pd.Series:
    """Resample to target frequency (period-end), taking last-of-period values.

    For monthly target with monthly source this is a no-op except for index
    alignment to month-end.
    """
    rule = {"M": "ME", "Q": "QE", "A": "YE"}[target_freq]
    return s.resample(rule).last().dropna()


def _flag_sustained_runs(boolean_series: pd.Series, min_length: int) -> pd.Series:
    """Flag True at periods within a True-run of length >= min_length.

    Example: with min_length=3,
        [F, T, T, T, F, T, T, F]  →  [F, T, T, T, F, F, F, F]

    Parameters
    ----------
    boolean_series : pd.Series of bool
    min_length : int

    Returns
    -------
    pd.Series of bool with same index
    """
    arr = boolean_series.to_numpy(dtype=bool)
    n = len(arr)
    out = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if arr[i]:
            j = i
            while j < n and arr[j]:
                j += 1
            run_len = j - i
            if run_len >= min_length:
                out[i:j] = True
            i = j
        else:
            i += 1

    return pd.Series(out, index=boolean_series.index)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_wicr_result(
    result: WicrResult,
    output_dir: str | Path,
    prefix: str = "wicr",
) -> dict[str, Path]:
    """Save WicrResult to CSV + JSON in output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.concat(
        [
            result.wage_growth_yoy,
            result.inflation_yoy,
            result.wicr_yoy,
            result.wicr_smoothed,
            result.regime_label,
            result.high_wicr_runs,
        ],
        axis=1,
    )
    panel_path = out / f"{prefix}_panel.csv"
    audit_path = out / f"{prefix}_audit.json"

    panel.to_csv(panel_path)
    with open(audit_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)

    return {"panel": panel_path, "audit": audit_path}
