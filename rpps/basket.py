"""
rpps.basket — the six-item fixed-quantity consumption basket.

Implements §4.3 of:
    Green, R. J. (2026). The Inflationary Yardstick. Working Paper.

The basket is intentionally simple, transparent, and constructible from public
data. It is NOT a substitute for the BLS CPI basket; it is a complement
designed to answer a different question — the labor-time question — which the
CPI is not designed to answer.

Composition (fixed physical quantities, NOT expenditure shares):

    Item            Quantity        Unit            Source series / data
    ─────────────── ─────────────── ─────────────── ────────────────────────────
    Gasoline        12              gallons/month   FRED GASREGW (1990+)
    Ground beef     40              lbs/month       FRED APU0000703112 (1984+)
    Tuition          1              year            NCES Digest of Education Stats
    Housing          1              median home     FRED MSPUS (1963+)
    Electricity   1000              kWh/month       FRED APU000074714 (1978+)
    Healthcare       1              year (family)   KFF Employer Health Benefits

Coverage limits: pre-1990 retail gasoline and pre-1984 ground beef are not
covered by the FRED series; the basket items report NaN for those dates and
the RPPH metric (Phase 2) excludes affected items from the composite by
masking. Robustness checks in Stage 2 will examine the sensitivity of the
composite RPPH to item inclusion/exclusion and basket re-weighting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rpps.fred_loader import REPO_ROOT, load_series

logger = logging.getLogger(__name__)


EXTERNAL_DIR = REPO_ROOT / "data" / "external"


# ---------------------------------------------------------------------------
# Item specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BasketItem:
    """Specification of a single basket item.

    Attributes
    ----------
    name : str
        Item identifier (used as column name in DataFrame outputs).
    quantity : float
        Fixed physical quantity (e.g. 12 gallons, 40 lbs, 1000 kWh).
    unit : str
        Physical unit ("gallon", "lb", "kWh", "year", "home").
    source : str
        Either a FRED series ID or one of the special tags:
        "external:nces_tuition" or "external:kff_healthcare".
    description : str
        Human-readable description.
    coverage_start_year : int
        First year for which item-level price data is available.
    notes : str
        Methodological notes for the item.
    """

    name: str
    quantity: float
    unit: str
    source: str
    description: str
    coverage_start_year: int
    notes: str = ""


BASKET_ITEMS: dict[str, BasketItem] = {
    "gasoline": BasketItem(
        name="gasoline",
        quantity=12.0,
        unit="gallon",
        source="GASREGW",
        description="Twelve gallons of regular gasoline (one driver, ~one month commute fuel)",
        coverage_start_year=1990,
        notes="Pre-1990 retail gasoline reconstruction available via EIA AER Table 5.24.",
    ),
    "beef": BasketItem(
        name="beef",
        quantity=40.0,
        unit="lb",
        source="APU0000703112",
        description="Forty pounds of ground beef, 100% beef (one month protein, family of four)",
        coverage_start_year=1984,
        notes="Pre-1984 retail ground beef from BLS CPI Average Price Data archive.",
    ),
    "tuition": BasketItem(
        name="tuition",
        quantity=1.0,
        unit="year",
        source="external:nces_tuition",
        description="One year, in-state public four-year university tuition + fees (sticker price)",
        coverage_start_year=1969,
        notes="NCES Digest of Education Statistics, Table 330.10. "
              "Pre-1969 reconstructed from individual-institution HEGIS records.",
    ),
    "housing": BasketItem(
        name="housing",
        quantity=1.0,
        unit="home",
        source="MSPUS",
        description="One median-priced new single-family home (single purchase value)",
        coverage_start_year=1963,
        notes="Mortgage-amortized cost computable separately given MORTGAGE30US.",
    ),
    "electricity": BasketItem(
        name="electricity",
        quantity=1000.0,
        unit="kWh",
        source="APU000074714",
        description="One thousand kWh of residential electricity (one month, typical household)",
        coverage_start_year=1978,
        notes="Pre-1978 retail electricity from EIA Annual Energy Review Table 8.10.",
    ),
    "healthcare": BasketItem(
        name="healthcare",
        quantity=1.0,
        unit="year",
        source="external:kff_healthcare",
        description="One year of typical employer-provided family health insurance "
                    "(total premium: employer + employee contributions)",
        coverage_start_year=1999,
        notes="KFF Employer Health Benefits Survey. Pre-1999 healthcare cost analysis "
              "is illustrative only, following Cutler and Meara (2001).",
    ),
}


# ---------------------------------------------------------------------------
# External data loaders
# ---------------------------------------------------------------------------

def load_nces_tuition() -> pd.Series:
    """Load the NCES tuition series (committed CSV under data/external/).

    Returns an annual Series (year-end timestamps) of total in-state public
    four-year tuition + required fees, in nominal dollars.
    """
    path = EXTERNAL_DIR / "nces_tuition.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"NCES tuition CSV not found at {path}. Expected committed file."
        )
    df = pd.read_csv(path, comment="#")
    required = {"year", "public_4yr_in_state_tuition_fees"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"NCES CSV must contain {required}. Found: {set(df.columns)}"
        )
    df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-12-31")
    series = df.set_index("date")["public_4yr_in_state_tuition_fees"].astype("float64")
    series.name = "nces_tuition_total"
    return series


def load_kff_healthcare() -> pd.Series:
    """Load the KFF healthcare series (committed CSV under data/external/).

    Returns an annual Series of total family health insurance premium (employer
    + employee contributions), in nominal dollars.
    """
    path = EXTERNAL_DIR / "kff_healthcare.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"KFF healthcare CSV not found at {path}. Expected committed file."
        )
    df = pd.read_csv(path, comment="#")
    required = {"year", "family_premium_total"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"KFF CSV must contain {required}. Found: {set(df.columns)}"
        )
    df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-12-31")
    series = df.set_index("date")["family_premium_total"].astype("float64")
    series.name = "kff_family_premium"
    return series


# ---------------------------------------------------------------------------
# Item price loading
# ---------------------------------------------------------------------------

def load_item_price(
    item: BasketItem,
    cache_dir: str | Path | None = None,
) -> pd.Series:
    """Load the unit-price series for a basket item.

    Returns a Series at the native frequency of the source. The series is the
    price per unit (e.g. price per gallon, per lb, per kWh, per year).
    """
    if item.source.startswith("external:"):
        tag = item.source.split(":", 1)[1]
        if tag == "nces_tuition":
            return load_nces_tuition()
        if tag == "kff_healthcare":
            return load_kff_healthcare()
        raise ValueError(f"Unknown external tag: {tag!r}")
    return load_series(item.source, cache_dir=cache_dir)


def item_cost(
    item: BasketItem,
    cache_dir: str | Path | None = None,
) -> pd.Series:
    """Return the nominal cost of one basket-quantity of the item over time.

    cost_t = quantity * price_t

    Example: for gasoline (quantity=12), cost_t is the dollar cost of 12
    gallons at the prevailing price.
    """
    price = load_item_price(item, cache_dir=cache_dir)
    cost = price * item.quantity
    cost.name = f"{item.name}_cost"
    return cost


# ---------------------------------------------------------------------------
# Composite basket
# ---------------------------------------------------------------------------

def basket_cost_panel(
    cache_dir: str | Path | None = None,
    items: list[str] | None = None,
    frequency: str = "M",
) -> pd.DataFrame:
    """Build a panel of nominal item costs at a chosen frequency.

    Parameters
    ----------
    cache_dir : optional cache directory override
    items : list of basket-item names to include. Default: all six items.
    frequency : pandas frequency string. Default "M" (month-end).

    Returns
    -------
    pd.DataFrame
        Index: pd.PeriodIndex or pd.DatetimeIndex aligned to the requested frequency.
        Columns: item names, values are nominal cost-of-quantity at that date.
    """
    target_items = items if items is not None else list(BASKET_ITEMS.keys())

    panels: dict[str, pd.Series] = {}
    for name in target_items:
        item = BASKET_ITEMS[name]
        try:
            cost = item_cost(item, cache_dir=cache_dir)
        except FileNotFoundError as exc:
            logger.warning("Item %s skipped (data not available): %s", name, exc)
            continue
        # Resample to monthly (forward-fill within month if source is annual or quarterly).
        # For annual series, we resample to monthly using forward-fill within a year.
        cost_resampled = _resample_to_frequency(cost, frequency)
        panels[name] = cost_resampled

    if not panels:
        return pd.DataFrame()

    df = pd.concat(panels, axis=1, sort=False)
    df.columns.name = "item"
    return df


def _resample_to_frequency(series: pd.Series, frequency: str) -> pd.Series:
    """Resample a price series to the requested frequency.

    For up-sampling (e.g. annual -> monthly), uses forward-fill within the
    sampling period. For down-sampling, uses the last value in the period.
    """
    target = frequency.upper()
    if target.startswith("M"):
        # Determine source frequency
        inferred = pd.infer_freq(series.index) or ""
        if inferred.startswith("A") or inferred.startswith("Y"):
            # Annual → monthly: forward-fill the annual value across the year
            monthly = series.resample("ME").last().ffill()
            return monthly
        if inferred.startswith("Q"):
            # Quarterly → monthly: forward-fill within the quarter
            monthly = series.resample("ME").last().ffill(limit=2)
            return monthly
        # Default: take last observation in each month
        return series.resample("ME").last()
    if target.startswith("Q"):
        return series.resample("QE").last()
    if target.startswith("A") or target.startswith("Y"):
        return series.resample("YE").last()
    raise ValueError(f"Unsupported frequency: {frequency}")


def basket_total_cost(
    cache_dir: str | Path | None = None,
    items: list[str] | None = None,
    frequency: str = "M",
    require_all_items: bool = False,
) -> pd.Series:
    """Total nominal cost of the basket at each date.

    Parameters
    ----------
    cache_dir : optional cache directory override
    items : list of basket-item names to include. Default: all six items.
    frequency : pandas frequency string.
    require_all_items : if True, the total is NaN for any date where any
        included item has NaN. If False, the total is the sum of available
        items at each date (used in early periods where some series are absent).

    Returns
    -------
    pd.Series
        Total basket cost in nominal dollars over time.
    """
    panel = basket_cost_panel(cache_dir=cache_dir, items=items, frequency=frequency)
    if panel.empty:
        return pd.Series(dtype="float64", name="basket_total_cost")

    if require_all_items:
        total = panel.sum(axis=1, skipna=False)
    else:
        total = panel.sum(axis=1, skipna=True, min_count=1)
    total.name = "basket_total_cost"
    return total


# ---------------------------------------------------------------------------
# Catalog inspection
# ---------------------------------------------------------------------------

def basket_summary() -> pd.DataFrame:
    """Return the basket specification as a DataFrame for inspection."""
    rows = [
        {
            "item": item.name,
            "quantity": item.quantity,
            "unit": item.unit,
            "source": item.source,
            "coverage_start_year": item.coverage_start_year,
            "description": item.description,
        }
        for item in BASKET_ITEMS.values()
    ]
    return pd.DataFrame(rows)
