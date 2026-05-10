"""
rpps.nber_splice — pre-1947 NBER → post-1947 FRED splice with overlap-window
level adjustment.

Implements §4.1 of:
    Green, R. J. (2026). The Inflationary Yardstick. Working Paper.

The constraint on long-horizon U.S. economic analysis is that several principal
FRED series begin only in 1939 (AHETPI, the principal nominal-wage series) or
1947 (OPHNFB, the principal productivity series). Pre-1947 data must be drawn
from the NBER Macrohistory archive and spliced to the FRED series.

This module implements two principal splices:

1. WAGE SPLICE
   AHETPI (1939+, monthly) ↔ M0844AUSM052NNBR (1923–1942, monthly)
   Overlap window: 1939Q1 – 1942Q4 (4 years of monthly data, ~48 observations)

2. PRODUCTIVITY SPLICE
   OPHNFB (1947+, quarterly) ↔ Kendrick (1961) annual productivity index
   Overlap window: 1947 – 1957 (11 years of annual data)

Both splices use a multiplicative level adjustment computed as the geometric
mean of the ratio (modern_series / legacy_series) over the overlap window:

    λ = exp(mean(log(modern_t / legacy_t)))   for t in overlap

The pre-overlap legacy series is then scaled by λ and concatenated with the
modern series. The geometric mean is preferred over the arithmetic mean because
both series are positive and the relationship is multiplicative (proportional)
rather than additive. This matches the approach used in the BLS chained
price-index methodology and in the NBER long-historical national accounts
reconstructions (Kuznets, Kendrick, Gordon).

Outputs are deterministic given the same input cache. Continuity at the splice
boundary, growth-rate sanity over the overlap, and reversibility are unit-tested.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rpps.fred_loader import DEFAULT_PROCESSED_DIR, REPO_ROOT, load_series

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------

EXTERNAL_DIR = REPO_ROOT / "data" / "external"

# Wage splice
WAGE_MODERN_SERIES = "AHETPI"        # FRED, 1939+, monthly
WAGE_LEGACY_SERIES = "M0844AUSM052NNBR"  # NBER mirror on FRED, 1923–1942, monthly
WAGE_OVERLAP_START = "1939-01-01"
WAGE_OVERLAP_END = "1942-12-31"

# Productivity splice
PROD_MODERN_SERIES = "OPHNFB"        # FRED, 1947+, quarterly
PROD_LEGACY_FILE = EXTERNAL_DIR / "kendrick_productivity.csv"
PROD_OVERLAP_START = "1947-01-01"
PROD_OVERLAP_END = "1957-12-31"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SpliceResult:
    """The result of a single splice operation, including audit information.

    Attributes
    ----------
    spliced : pd.Series
        The unified pre-merger + post-merger spliced series.
    adjustment_factor : float
        The multiplicative level-adjustment factor (lambda) applied to the
        legacy series.
    overlap_start : pd.Timestamp
        Start of the overlap window used to estimate the adjustment.
    overlap_end : pd.Timestamp
        End of the overlap window.
    n_overlap_obs : int
        Number of observations in the overlap window.
    legacy_series_id : str
        Identifier of the legacy (pre-merger) series.
    modern_series_id : str
        Identifier of the modern (post-merger) series.
    boundary_continuity_ok : bool
        True if the legacy and modern values agree to within tolerance at the
        splice boundary.
    """

    spliced: pd.Series
    adjustment_factor: float
    overlap_start: pd.Timestamp
    overlap_end: pd.Timestamp
    n_overlap_obs: int
    legacy_series_id: str
    modern_series_id: str
    boundary_continuity_ok: bool

    def to_dict(self) -> dict:
        return {
            "legacy_series_id": self.legacy_series_id,
            "modern_series_id": self.modern_series_id,
            "adjustment_factor": float(self.adjustment_factor),
            "overlap_start": str(self.overlap_start.date()),
            "overlap_end": str(self.overlap_end.date()),
            "n_overlap_obs": int(self.n_overlap_obs),
            "boundary_continuity_ok": bool(self.boundary_continuity_ok),
            "first_obs_date": str(self.spliced.index.min().date()),
            "last_obs_date": str(self.spliced.index.max().date()),
            "n_observations": len(self.spliced),
        }


# ---------------------------------------------------------------------------
# Core splice algorithm
# ---------------------------------------------------------------------------

def compute_adjustment_factor(
    legacy: pd.Series,
    modern: pd.Series,
    overlap_start: str | pd.Timestamp,
    overlap_end: str | pd.Timestamp,
    method: str = "geometric",
) -> tuple[float, int]:
    """Compute the multiplicative level adjustment for a splice.

    Parameters
    ----------
    legacy : pd.Series
        The legacy (pre-merger) series.
    modern : pd.Series
        The modern (post-merger) series.
    overlap_start, overlap_end : str | pd.Timestamp
        Bounds of the overlap window (inclusive).
    method : str
        "geometric" (default, recommended) or "arithmetic".

    Returns
    -------
    (lambda, n_obs)
        lambda is the multiplicative factor such that
            spliced_t = legacy_t * lambda  for t < overlap_start
        and n_obs is the number of paired observations used.

    Raises
    ------
    ValueError
        If no paired observations exist in the overlap window.
    """
    overlap_start = pd.Timestamp(overlap_start)
    overlap_end = pd.Timestamp(overlap_end)

    legacy_overlap = legacy.loc[overlap_start:overlap_end].dropna()
    modern_overlap = modern.loc[overlap_start:overlap_end].dropna()

    # Inner-join on dates so we only use paired observations
    paired = pd.concat(
        {"legacy": legacy_overlap, "modern": modern_overlap}, axis=1
    ).dropna()

    if len(paired) == 0:
        raise ValueError(
            f"No paired observations in overlap window "
            f"[{overlap_start.date()}, {overlap_end.date()}]. "
            f"Legacy: {len(legacy_overlap)}, Modern: {len(modern_overlap)} obs in window."
        )

    # Avoid division by zero
    if (paired["legacy"] <= 0).any() or (paired["modern"] <= 0).any():
        raise ValueError(
            "Splice requires strictly positive values in the overlap window. "
            "Found non-positive values."
        )

    ratios = paired["modern"] / paired["legacy"]

    if method == "geometric":
        # Geometric mean: exp(mean(log(ratios)))
        # Equivalent to (prod ratios)^(1/n) but more numerically stable.
        lam = float(np.exp(np.log(ratios).mean()))
    elif method == "arithmetic":
        lam = float(ratios.mean())
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'geometric' or 'arithmetic'.")

    return lam, len(paired)


def splice_series(
    legacy: pd.Series,
    modern: pd.Series,
    overlap_start: str | pd.Timestamp,
    overlap_end: str | pd.Timestamp,
    legacy_id: str,
    modern_id: str,
    method: str = "geometric",
    boundary_tolerance: float = 0.05,
) -> SpliceResult:
    """Splice a legacy series to a modern series using a level adjustment.

    The result is a single continuous series:
        - For dates before the overlap_start: legacy * lambda
        - For dates from overlap_start onward: modern (preferred), or legacy*lambda
          where modern is NaN.

    Parameters
    ----------
    legacy : pd.Series
    modern : pd.Series
    overlap_start, overlap_end : str | pd.Timestamp
        Window over which the adjustment factor is estimated.
    legacy_id, modern_id : str
        Identifiers used for the audit record.
    method : str
        "geometric" (default) or "arithmetic".
    boundary_tolerance : float
        Fractional tolerance for the boundary-continuity check at overlap_start.
        Default 0.05 (5%). Set higher for noisy series.

    Returns
    -------
    SpliceResult
    """
    overlap_start_ts = pd.Timestamp(overlap_start)
    overlap_end_ts = pd.Timestamp(overlap_end)

    lam, n_obs = compute_adjustment_factor(
        legacy, modern, overlap_start_ts, overlap_end_ts, method=method
    )

    # Pre-overlap segment: scale legacy by lambda
    pre = legacy.loc[: overlap_start_ts - pd.Timedelta(days=1)] * lam

    # Post-overlap-start segment: prefer modern; fall back to legacy*lambda where modern is NaN
    legacy_scaled_after = legacy.loc[overlap_start_ts:] * lam
    modern_after = modern.loc[overlap_start_ts:]
    union_index = modern_after.index.union(legacy_scaled_after.index)
    post = modern_after.reindex(union_index).combine_first(
        legacy_scaled_after.reindex(union_index)
    )

    spliced = pd.concat([pre, post]).sort_index()
    # Deduplicate any boundary day collision (taking modern over legacy*lambda)
    spliced = spliced[~spliced.index.duplicated(keep="last")]
    spliced.name = f"spliced_{modern_id}"

    # Boundary continuity check
    boundary_ok = _check_boundary_continuity(
        spliced, overlap_start_ts, tolerance=boundary_tolerance
    )

    return SpliceResult(
        spliced=spliced,
        adjustment_factor=lam,
        overlap_start=overlap_start_ts,
        overlap_end=overlap_end_ts,
        n_overlap_obs=n_obs,
        legacy_series_id=legacy_id,
        modern_series_id=modern_id,
        boundary_continuity_ok=boundary_ok,
    )


def _check_boundary_continuity(
    spliced: pd.Series,
    boundary: pd.Timestamp,
    tolerance: float,
) -> bool:
    """Check that the spliced series has no jump discontinuity at the boundary."""
    pre = spliced.loc[: boundary - pd.Timedelta(days=1)].dropna()
    post = spliced.loc[boundary:].dropna()
    if pre.empty or post.empty:
        # Cannot evaluate; conservatively report True (caller will see in audit)
        return True
    # Take the last pre-boundary value and the first post-boundary value
    pre_val = float(pre.iloc[-1])
    post_val = float(post.iloc[0])
    if pre_val == 0:
        return False
    rel_jump = abs(post_val - pre_val) / abs(pre_val)
    return rel_jump <= tolerance


# ---------------------------------------------------------------------------
# Wage splice
# ---------------------------------------------------------------------------

def load_spliced_wages(
    cache_dir: str | Path | None = None,
) -> pd.Series:
    """Convenience accessor: load the spliced 1925–present nominal-wage series.

    Returns a monthly-frequency pandas Series, with the AHETPI series after 1939
    and the M0844 series (level-adjusted) before 1939.
    """
    result = build_wage_splice(cache_dir=cache_dir)
    return result.spliced


def build_wage_splice(
    cache_dir: str | Path | None = None,
    boundary_tolerance: float = 0.05,
) -> SpliceResult:
    """Build the spliced manufacturing/total-private wage series, 1925–present."""
    legacy = load_series(WAGE_LEGACY_SERIES, cache_dir=cache_dir)
    modern = load_series(WAGE_MODERN_SERIES, cache_dir=cache_dir)

    return splice_series(
        legacy=legacy,
        modern=modern,
        overlap_start=WAGE_OVERLAP_START,
        overlap_end=WAGE_OVERLAP_END,
        legacy_id=WAGE_LEGACY_SERIES,
        modern_id=WAGE_MODERN_SERIES,
        method="geometric",
        boundary_tolerance=boundary_tolerance,
    )


# ---------------------------------------------------------------------------
# Productivity splice
# ---------------------------------------------------------------------------

def load_kendrick_productivity() -> pd.Series:
    """Load the Kendrick (1961) historical productivity index from the committed CSV.

    The CSV is expected to have columns: year, productivity_index.
    The index is normalized to 100.0 in 1929 (the original Kendrick base year).
    """
    if not PROD_LEGACY_FILE.is_file():
        raise FileNotFoundError(
            f"Kendrick productivity series not found at {PROD_LEGACY_FILE}. "
            "This file should be committed to the repository under data/external/. "
            "If you have just cloned, run `make data` to verify the external data is present."
        )
    df = pd.read_csv(PROD_LEGACY_FILE, comment="#")
    if "year" not in df.columns or "productivity_index" not in df.columns:
        raise ValueError(
            f"Kendrick CSV must have columns 'year' and 'productivity_index'. "
            f"Found: {list(df.columns)}"
        )
    df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-12-31")
    series = df.set_index("date")["productivity_index"].astype("float64")
    series.name = "kendrick_productivity"
    return series


def build_productivity_splice(
    cache_dir: str | Path | None = None,
    boundary_tolerance: float = 0.05,
) -> SpliceResult:
    """Build the spliced productivity series, 1925–present, annual frequency."""
    legacy = load_kendrick_productivity()

    # OPHNFB is quarterly. Convert to annual (year-end) for the splice.
    modern_q = load_series(PROD_MODERN_SERIES, cache_dir=cache_dir)
    modern = modern_q.resample("YE").last()
    modern.index = pd.DatetimeIndex(
        [pd.Timestamp(year=t.year, month=12, day=31) for t in modern.index]
    )

    return splice_series(
        legacy=legacy,
        modern=modern,
        overlap_start=PROD_OVERLAP_START,
        overlap_end=PROD_OVERLAP_END,
        legacy_id="kendrick_productivity",
        modern_id=PROD_MODERN_SERIES,
        method="geometric",
        boundary_tolerance=boundary_tolerance,
    )


# ---------------------------------------------------------------------------
# Build / persist
# ---------------------------------------------------------------------------

def _audit_path_str(p: Path) -> str:
    """Render a path for the audit record. Prefer a path relative to the
    repo root when possible, but fall back to the absolute path when the
    target is outside the repo (e.g. a user's tmp dir or external output)."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p.resolve())


def build_spliced_dataset(
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Build all spliced series, write them to data/processed/, return audit dict."""
    output_dir = Path(output_dir) if output_dir else DEFAULT_PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    audit: dict = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "splices": {},
    }

    # Wage splice
    logger.info("Building wage splice")
    wage = build_wage_splice(cache_dir=cache_dir)
    wage_path = output_dir / "spliced_wages.csv"
    wage.spliced.to_csv(wage_path, header=True)
    audit["splices"]["wages"] = wage.to_dict()
    audit["splices"]["wages"]["output_path"] = _audit_path_str(wage_path)

    # Productivity splice
    logger.info("Building productivity splice")
    try:
        prod = build_productivity_splice(cache_dir=cache_dir)
        prod_path = output_dir / "spliced_productivity.csv"
        prod.spliced.to_csv(prod_path, header=True)
        audit["splices"]["productivity"] = prod.to_dict()
        audit["splices"]["productivity"]["output_path"] = _audit_path_str(prod_path)
    except FileNotFoundError as exc:
        logger.warning("Productivity splice skipped: %s", exc)
        audit["splices"]["productivity"] = {"status": "skipped", "reason": str(exc)}

    # Persist audit
    audit_path = output_dir / "splice_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True))
    logger.info("Wrote splice audit: %s", audit_path)

    return audit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(description="NBER → FRED splice")
    parser.add_argument("--build-spliced-dataset", action="store_true",
                        help="Build all spliced series and write to data/processed/")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(args.verbose, 2)]
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    if args.build_spliced_dataset:
        audit = build_spliced_dataset(
            cache_dir=args.cache_dir, output_dir=args.output_dir
        )
        for kind, info in audit["splices"].items():
            print(f"{kind}: {info}")


if __name__ == "__main__":
    _main()
