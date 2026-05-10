"""
rpps.metrics.prwdi — Productivity-Real-Wage Decoupling Index.

Implements §5.1, metric #3 of:
    Green, R. J. (2026). The Inflationary Yardstick. Working Paper.

Definition
----------
For productivity index Q_t (BLS OPHNFB for 1947+, Kendrick (1961) historical
extension for 1925-1947) and real-compensation index C_t (BLS COMPRNFB for
1947+, spliced manufacturing wage / CPI for 1925-1947), normalized to a
common base year:

    PRWDI_t = (Q_t / Q_base) / (C_t / C_base)

Default base = 1947 (the start of the consistent post-war BLS productivity
program). The Mishel-Bivens (2015) productivity-pay decoupling implies PRWDI
rising sharply after 1973. Pre-1947 PRWDI uses Kendrick historical productivity
and is reported with explicit measurement-uncertainty caveats per §4.4.

Interpretation
--------------
    PRWDI = 1.00  :  productivity and real compensation moved together since base year
    PRWDI > 1.00  :  productivity outpaced real compensation (decoupling); workers received
                     a smaller share of productivity gains
    PRWDI < 1.00  :  real compensation outpaced productivity (rare; transfer or terms-of-trade
                     shifts)

The metric is dimensionless and is the cumulative ratio. The annual change
ΔPRWDI captures the within-period decoupling rate.

Connection to Stansbury-Summers (2018)
--------------------------------------
The Stansbury-Summers critique argues that some of the apparent decoupling
reflects different deflators applied to output (producer prices) vs.
compensation (consumer prices). The PRWDI as computed here uses the BLS
real-compensation series COMPRNFB, which is consumer-price-deflated; a
matched-deflator robustness check in §6.4 substitutes producer-price-deflated
real compensation and reports the resulting PRWDI alongside the principal
specification.

Usage
-----
    from rpps.metrics.prwdi import compute_prwdi
    result = compute_prwdi(productivity_series, real_compensation_series, base_year=1947)
    result.prwdi              # pd.Series, cumulative decoupling ratio
    result.delta_prwdi_annual # pd.Series, annual change
    result.audit
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_BASE_YEAR = 1947


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PrwdiResult:
    """Container for Productivity-Real-Wage Decoupling Index computation.

    Attributes
    ----------
    prwdi : pd.Series
        PRWDI_t = (Q_t/Q_base) / (C_t/C_base). Equal to 1.0 at base year.
    productivity_index : pd.Series
        Q_t / Q_base (productivity normalized to base year).
    compensation_index : pd.Series
        C_t / C_base (compensation normalized to base year).
    delta_prwdi_annual : pd.Series
        Year-over-year change in PRWDI (annualized for sub-annual data).
    base_year : int
    n_observations : int
    coverage_start : pd.Timestamp | None
    coverage_end : pd.Timestamp | None
    audit : dict
    """

    prwdi: pd.Series
    productivity_index: pd.Series
    compensation_index: pd.Series
    delta_prwdi_annual: pd.Series
    base_year: int
    n_observations: int
    coverage_start: pd.Timestamp | None
    coverage_end: pd.Timestamp | None
    audit: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "base_year": self.base_year,
            "n_observations": self.n_observations,
            "coverage_start": self.coverage_start.isoformat() if self.coverage_start is not None else None,
            "coverage_end": self.coverage_end.isoformat() if self.coverage_end is not None else None,
            "prwdi_at_end": float(self.prwdi.dropna().iloc[-1]) if self.n_observations else None,
            "audit": self.audit,
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_prwdi(
    productivity: pd.Series,
    real_compensation: pd.Series,
    *,
    base_year: int = DEFAULT_BASE_YEAR,
    productivity_id: str = "productivity",
    compensation_id: str = "real_compensation",
) -> PrwdiResult:
    """Compute the Productivity-Real-Wage Decoupling Index.

    Parameters
    ----------
    productivity : pd.Series
        Productivity index or level series (e.g. OPHNFB; output per hour).
    real_compensation : pd.Series
        Real compensation per hour (e.g. COMPRNFB).
    base_year : int, default 1947
        Year against which both series are normalized to 1.00.
    productivity_id, compensation_id : str
        Identifiers recorded in audit.

    Returns
    -------
    PrwdiResult
    """
    if productivity.empty:
        raise ValueError("productivity series is empty")
    if real_compensation.empty:
        raise ValueError("real_compensation series is empty")

    prod = _to_datetime_index(productivity).sort_index()
    comp = _to_datetime_index(real_compensation).sort_index()

    # Find base-year values. We use the first observation in the calendar
    # year matching base_year, falling back to the closest available period
    # within ±1 year if base_year is not present.
    prod_base = _base_value(prod, base_year)
    comp_base = _base_value(comp, base_year)

    if prod_base is None:
        raise ValueError(
            f"productivity series has no observation in or near base_year={base_year}"
        )
    if comp_base is None:
        raise ValueError(
            f"real_compensation series has no observation in or near base_year={base_year}"
        )

    prod_index = prod / prod_base
    comp_index = comp / comp_base
    prod_index.name = "productivity_index"
    comp_index.name = "compensation_index"

    # Align on common index (both series resampled to the more-granular freq).
    common_idx = prod_index.index.intersection(comp_index.index)
    if len(common_idx) < 2:
        raise ValueError("insufficient overlap between productivity and compensation")

    prwdi = (prod_index.loc[common_idx]) / (comp_index.loc[common_idx])
    prwdi.name = "prwdi"

    # Annual change. For quarterly data we use 4-quarter lag; for annual we use 1.
    annual_lag = _annual_lag(prwdi.index)
    delta = prwdi.pct_change(periods=annual_lag)
    delta.name = "delta_prwdi_annual"

    valid = prwdi.dropna()
    audit = {
        "computation": "PRWDI = (Q/Q_base) / (C/C_base)",
        "base_year": base_year,
        "productivity_base_value": float(prod_base),
        "compensation_base_value": float(comp_base),
        "productivity_id": productivity_id,
        "compensation_id": compensation_id,
        "annual_lag_periods": annual_lag,
        "productivity_n_obs": len(productivity),
        "compensation_n_obs": len(real_compensation),
        "joint_n_obs": len(common_idx),
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return PrwdiResult(
        prwdi=prwdi,
        productivity_index=prod_index.loc[common_idx],
        compensation_index=comp_index.loc[common_idx],
        delta_prwdi_annual=delta,
        base_year=base_year,
        n_observations=len(valid),
        coverage_start=valid.index.min() if len(valid) else None,
        coverage_end=valid.index.max() if len(valid) else None,
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


def _base_value(series: pd.Series, base_year: int) -> float | None:
    """Return the mean value over the base year, or nearest-year fallback."""
    in_year = series[series.index.year == base_year]
    if not in_year.empty:
        return float(in_year.mean())
    # Fallback to ±1 year window.
    nearby = series[(series.index.year >= base_year - 1) & (series.index.year <= base_year + 1)]
    if not nearby.empty:
        logger.warning(
            "Base year %d not found exactly; using %d +/- 1 window mean.",
            base_year,
            base_year,
        )
        return float(nearby.mean())
    return None


def _annual_lag(idx: pd.DatetimeIndex) -> int:
    """Number of periods in one year for the given index frequency."""
    if len(idx) < 2:
        return 1
    diffs = pd.Series(idx).diff().dropna()
    median_days = diffs.dt.days.median()
    if median_days <= 35:
        return 12  # monthly
    if median_days <= 100:
        return 4   # quarterly
    return 1       # annual or coarser


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_prwdi_result(
    result: PrwdiResult,
    output_dir: str | Path,
    prefix: str = "prwdi",
) -> dict[str, Path]:
    """Save PrwdiResult to CSV + JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.concat(
        [result.productivity_index, result.compensation_index, result.prwdi, result.delta_prwdi_annual],
        axis=1,
    )
    panel_path = out / f"{prefix}_panel.csv"
    audit_path = out / f"{prefix}_audit.json"

    panel.to_csv(panel_path)
    with open(audit_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)

    return {"panel": panel_path, "audit": audit_path}
