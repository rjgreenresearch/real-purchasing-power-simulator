"""
rpps.metrics.rpph — Real Purchasing Power Hours.

Implements §5.1, metric #1 of:
    Green, R. J. (2026). The Inflationary Yardstick. Working Paper.

Definition
----------
For basket item i with nominal price p_{i,t} at time t, fixed physical quantity
q_i, and reference wage w_t (Average Hourly Earnings, AHETPI for 1939+, spliced
manufacturing wage for 1925-1939):

    RPPH_{i,t}  = (q_i * p_{i,t}) / w_t      (item-level, hours)
    RPPH_t      = sum_i RPPH_{i,t}           (composite, hours)

The metric answers the question: how many hours of labor at the prevailing
median production-worker wage are required to purchase the fixed basket?
Falling RPPH indicates rising real purchasing power. Rising RPPH indicates
falling real purchasing power.

The metric is denominated in hours, not dollars, deliberately. Hours are
invariant to nominal regime changes, monetary redenominations, and price-level
shifts that the conventional CPI-deflated real-wage statistic absorbs into its
deflator. The labor-time accounting is the welfare-relevant magnitude under
the framework developed in §3.

Usage
-----
    from rpps.metrics.rpph import compute_rpph
    result = compute_rpph(basket_panel, wage_series)
    result.composite   # pd.Series, RPPH hours over time
    result.by_item     # pd.DataFrame, per-item RPPH
    result.audit       # dict, traceability metadata
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RpphResult:
    """Container for Real Purchasing Power Hours computation.

    Attributes
    ----------
    composite : pd.Series
        Total basket RPPH over time (hours of labor to buy the full basket).
        NaN where any included item lacks data.
    by_item : pd.DataFrame
        Per-item RPPH. Columns are item names; values are item-level hours.
        NaN where source data is missing for that item.
    items_used : list[str]
        Names of basket items included in the composite.
    wage_series_id : str
        Identifier of the wage series used (e.g. "AHETPI" or "wage_spliced").
    n_observations : int
        Number of non-NaN composite observations.
    coverage_start : pd.Timestamp | None
        First date with a valid composite value.
    coverage_end : pd.Timestamp | None
        Last date with a valid composite value.
    audit : dict
        Computation metadata for reproducibility.
    """

    composite: pd.Series
    by_item: pd.DataFrame
    items_used: list[str]
    wage_series_id: str
    n_observations: int
    coverage_start: pd.Timestamp | None
    coverage_end: pd.Timestamp | None
    audit: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serializable summary."""
        return {
            "wage_series_id": self.wage_series_id,
            "items_used": self.items_used,
            "n_observations": self.n_observations,
            "coverage_start": self.coverage_start.isoformat() if self.coverage_start is not None else None,
            "coverage_end": self.coverage_end.isoformat() if self.coverage_end is not None else None,
            "audit": self.audit,
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_rpph(
    basket_panel: pd.DataFrame,
    wage_series: pd.Series,
    *,
    require_all_items: bool = False,
    wage_series_id: str | None = None,
) -> RpphResult:
    """Compute Real Purchasing Power Hours from basket and wage data.

    Parameters
    ----------
    basket_panel : pd.DataFrame
        Output of `rpps.basket.basket_cost_panel`. Index is a DatetimeIndex
        at consistent frequency. Columns are basket items, values are nominal
        cost-of-quantity in dollars.
    wage_series : pd.Series
        Reference hourly wage series at consistent frequency. Index must be
        alignable with basket_panel via reindex/forward-fill.
    require_all_items : bool, default False
        If True, raise ValueError if any panel item has missing values within
        the joint coverage period. If False (default), composite is computed
        as sum-of-available-items, with periods where any item is NaN producing
        a NaN composite (sum-with-NaN-propagation behavior).
    wage_series_id : str | None
        Identifier for the wage series, recorded in the audit trail. If None,
        uses wage_series.name or "unknown".

    Returns
    -------
    RpphResult
        Container with composite, per-item, and audit metadata.

    Raises
    ------
    ValueError
        If basket_panel is empty, or if require_all_items=True and panel
        contains gaps within the joint coverage period.
    """
    if basket_panel.empty:
        raise ValueError("basket_panel is empty; nothing to compute.")
    if wage_series.empty:
        raise ValueError("wage_series is empty; cannot compute RPPH.")

    wage_id = wage_series_id or wage_series.name or "unknown"

    # Align wage to basket frequency.
    aligned_wage = _align_wage_to_panel(wage_series, basket_panel)

    # Per-item RPPH: nominal cost / hourly wage = hours.
    by_item = basket_panel.div(aligned_wage, axis=0)

    # Composite: sum across items.
    if require_all_items:
        if by_item.isna().any().any():
            missing = by_item.columns[by_item.isna().any()].tolist()
            raise ValueError(
                f"require_all_items=True but missing values in items: {missing}"
            )
        composite = by_item.sum(axis=1)
    else:
        # Sum with min_count=1 means rows with all-NaN stay NaN; rows with
        # any valid value get the sum-of-valid. We instead use strict
        # sum-with-NaN-propagation: composite NaN if ANY item NaN, because
        # composite means the FULL basket. Item-level RPPH is available
        # via by_item for partial-basket analysis.
        composite = by_item.sum(axis=1, min_count=len(by_item.columns))

    composite.name = "rpph_composite_hours"

    valid = composite.dropna()
    n_obs = len(valid)
    coverage_start = valid.index.min() if n_obs else None
    coverage_end = valid.index.max() if n_obs else None

    audit = {
        "computation": "RPPH = sum(item_cost) / hourly_wage",
        "items": list(basket_panel.columns),
        "wage_series_id": str(wage_id),
        "panel_index_freq": _safe_freq(basket_panel.index),
        "wage_index_freq": _safe_freq(wage_series.index),
        "require_all_items": require_all_items,
        "panel_n_rows": len(basket_panel),
        "wage_n_rows": len(wage_series),
        "composite_n_obs": n_obs,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return RpphResult(
        composite=composite,
        by_item=by_item,
        items_used=list(basket_panel.columns),
        wage_series_id=str(wage_id),
        n_observations=n_obs,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        audit=audit,
    )


def _align_wage_to_panel(
    wage: pd.Series,
    panel: pd.DataFrame,
) -> pd.Series:
    """Reindex wage to the panel's index, forward-filling within reasonable gaps.

    Wage data is typically monthly; basket panel may be monthly, quarterly, or
    annual depending on caller choice. We reindex with method='ffill' and a
    tolerance of ~93 days to avoid filling across multi-year gaps.
    """
    if not isinstance(panel.index, pd.DatetimeIndex):
        panel_dt = pd.DatetimeIndex(panel.index)
    else:
        panel_dt = panel.index

    if not isinstance(wage.index, pd.DatetimeIndex):
        wage = wage.copy()
        wage.index = pd.DatetimeIndex(wage.index)

    wage_sorted = wage.sort_index()
    aligned = wage_sorted.reindex(
        panel_dt,
        method="ffill",
        tolerance=pd.Timedelta(days=93),
    )
    aligned.index = panel.index
    return aligned


def _safe_freq(idx: pd.Index) -> str | None:
    """Return the inferred frequency of an index, or None if not detectable."""
    if not isinstance(idx, pd.DatetimeIndex):
        return None
    try:
        return idx.freqstr or pd.infer_freq(idx)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Item-level convenience
# ---------------------------------------------------------------------------

def labor_hours_for_item(
    item_cost: pd.Series,
    wage_series: pd.Series,
) -> pd.Series:
    """Convenience wrapper: hours of labor to buy a specific item over time.

    Parameters
    ----------
    item_cost : pd.Series
        Nominal cost of the item (e.g. 12 gallons of gasoline, or one home).
    wage_series : pd.Series
        Hourly wage series.

    Returns
    -------
    pd.Series
        Hours required at the prevailing wage.
    """
    aligned_wage = wage_series.reindex(
        item_cost.index, method="ffill", tolerance=pd.Timedelta(days=93)
    )
    hours = item_cost / aligned_wage
    hours.name = f"{item_cost.name}_hours" if item_cost.name else "hours"
    return hours


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_rpph_result(
    result: RpphResult,
    output_dir: str | Path,
    prefix: str = "rpph",
) -> dict[str, Path]:
    """Save RpphResult to CSV + JSON in output_dir. Returns paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    composite_path = out / f"{prefix}_composite.csv"
    by_item_path = out / f"{prefix}_by_item.csv"
    audit_path = out / f"{prefix}_audit.json"

    result.composite.to_csv(composite_path, header=True)
    result.by_item.to_csv(by_item_path)
    with open(audit_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)

    return {
        "composite": composite_path,
        "by_item": by_item_path,
        "audit": audit_path,
    }
