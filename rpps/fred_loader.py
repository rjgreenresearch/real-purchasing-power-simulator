"""
rpps.fred_loader — FRED API data acquisition and caching.

Implements §4.2 and Appendix A of:
    Green, R. J. (2026). The Inflationary Yardstick. Working Paper.

Provides:
    - FRED_SERIES catalog (all series IDs, descriptions, frequencies, coverage)
    - download_series()    fetch one series from the FRED API with caching
    - load_series()        load a cached series as a pandas Series
    - download_all()       batch-download every series in the catalog
    - build_manifest()     write the data/processed/manifest.json audit trail

The cache is keyed by FRED series ID and stored as CSV under data/raw/fred/.
The cache is invalidated by the --force flag or by deleting the cache directory.
A free FRED API key is required: https://fred.stlouisfed.org/docs/api/api_key.html

Tests for this module use fixture data and do not require a network connection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES_INFO_URL = "https://api.stlouisfed.org/fred/series"
FRED_RATE_LIMIT_PER_MINUTE = 120
FRED_RATE_LIMIT_SLEEP = 60.0 / FRED_RATE_LIMIT_PER_MINUTE  # ~0.5 s between calls

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "raw" / "fred"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Series catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FredSeries:
    """Metadata for a single FRED series tracked by the simulator."""

    id: str
    description: str
    frequency: str          # "M", "Q", "A", "W", "D"
    start_year: int
    category: str           # "price", "wage", "productivity", "household", "asset", "energy"
    notes: str = ""


# Catalog organized by analytical category.
# Coverage start_year is the year the series first becomes available on FRED.
# Categories correspond to Appendix A.1–A.6 of the paper.

FRED_SERIES: dict[str, FredSeries] = {
    # ------------------------------------------------------------------ Prices (A.1)
    "CPIAUCNS": FredSeries(
        "CPIAUCNS", "CPI for All Urban Consumers: All Items, NSA", "M", 1913, "price",
        notes="Primary deflator for nominal wages."),
    "CPIAUCSL": FredSeries(
        "CPIAUCSL", "CPI for All Urban Consumers: All Items, SA", "M", 1947, "price"),
    "PPIACO": FredSeries(
        "PPIACO", "Producer Price Index by Commodity: All Commodities", "M", 1913, "price",
        notes="Primary metric for input-cost trajectory."),
    "GDPDEF": FredSeries(
        "GDPDEF", "GDP Implicit Price Deflator", "Q", 1947, "price",
        notes="Cross-check deflator following Stansbury & Summers (2018)."),
    "PCEPI": FredSeries(
        "PCEPI", "PCE Chain-Type Price Index", "M", 1959, "price"),
    "WPS0561": FredSeries(
        "WPS0561", "PPI by Commodity: Crude Petroleum (Domestic Production)", "M", 1913, "price"),
    "WPU0911": FredSeries(
        "WPU0911", "PPI by Commodity: Pulp, Paper, and Allied Products", "M", 1947, "price"),
    "WPU051": FredSeries(
        "WPU051", "PPI by Commodity: Coal", "M", 1958, "price"),
    "WPU101": FredSeries(
        "WPU101", "PPI by Commodity: Iron and Steel", "M", 1926, "price"),

    # ------------------------------------------------------------------ Wages (A.2)
    "AHETPI": FredSeries(
        "AHETPI", "Avg Hourly Earnings of Production and Nonsupervisory Employees", "M", 1939, "wage",
        notes="Principal nominal wage series. Spliced with M0844 pre-1939."),
    "CES0500000003": FredSeries(
        "CES0500000003", "Avg Hourly Earnings of All Employees: Total Private", "M", 2006, "wage"),
    "COMPRNFB": FredSeries(
        "COMPRNFB", "Real Compensation per Hour, Nonfarm Business", "Q", 1947, "wage",
        notes="Used in PRWDI metric and counterfactual."),
    "COMPNFB": FredSeries(
        "COMPNFB", "Compensation per Hour, Nonfarm Business (nominal)", "Q", 1947, "wage"),
    "LES1252881600Q": FredSeries(
        "LES1252881600Q", "Median Usual Weekly Real Earnings", "Q", 1979, "wage"),
    "MEHOINUSA672N": FredSeries(
        "MEHOINUSA672N", "Real Median Household Income", "A", 1984, "wage"),
    "MEPAINUSA672N": FredSeries(
        "MEPAINUSA672N", "Real Mean Household Income", "A", 1984, "wage"),
    "DSPIC96": FredSeries(
        "DSPIC96", "Real Disposable Personal Income", "M", 1959, "wage"),
    "A229RX0": FredSeries(
        "A229RX0", "Real Disposable Personal Income: Per Capita", "M", 1959, "wage"),
    "PRS85006023": FredSeries(
        "PRS85006023", "Nonfarm Business Sector: Hours Worked", "Q", 1947, "wage"),

    # ------------------------------------------------------------------ Productivity (A.3)
    "OPHNFB": FredSeries(
        "OPHNFB", "Nonfarm Business Sector: Real Output Per Hour", "Q", 1947, "productivity",
        notes="Principal productivity series. Spliced with Kendrick pre-1947."),
    "MFGOPH": FredSeries(
        "MFGOPH", "Manufacturing Sector: Real Output Per Hour", "Q", 1987, "productivity"),
    "ULCNFB": FredSeries(
        "ULCNFB", "Nonfarm Business Sector: Unit Labor Cost", "Q", 1947, "productivity"),
    "LABSHPUSA156NRUG": FredSeries(
        "LABSHPUSA156NRUG", "Share of Labor Compensation in GDP", "A", 1950, "productivity"),
    "PRS85006092": FredSeries(
        "PRS85006092", "Nonfarm Business Sector: Implicit Price Deflator", "Q", 1947, "productivity"),

    # ------------------------------------------------------------------ Household (A.4)
    "PSAVERT": FredSeries(
        "PSAVERT", "Personal Saving Rate", "M", 1959, "household"),
    "TDSP": FredSeries(
        "TDSP", "Household Debt Service Payments / DPI", "Q", 1980, "household"),
    "FODSP": FredSeries(
        "FODSP", "Financial Obligations Ratio for Households", "Q", 1980, "household"),
    "TOTALSL": FredSeries(
        "TOTALSL", "Total Consumer Credit Owned and Securitized", "M", 1943, "household"),
    "REVOLSL": FredSeries(
        "REVOLSL", "Revolving Consumer Credit", "M", 1968, "household"),
    "DRALACBN": FredSeries(
        "DRALACBN", "Delinquency Rate, All Loans, All Commercial Banks", "Q", 1985, "household"),
    "DRSFRMACBS": FredSeries(
        "DRSFRMACBS", "Delinquency Rate, Single-Family Residential Mortgages", "Q", 1991, "household"),

    # ------------------------------------------------------------------ Assets (A.5)
    "CSUSHPISA": FredSeries(
        "CSUSHPISA", "S&P/Case-Shiller National Home Price Index", "M", 1987, "asset"),
    "MSPUS": FredSeries(
        "MSPUS", "Median Sales Price of Houses Sold", "Q", 1963, "asset",
        notes="Used in housing item of consumption basket."),
    "ASPUS": FredSeries(
        "ASPUS", "Average Sales Price of Houses Sold", "Q", 1963, "asset"),
    "SP500": FredSeries(
        "SP500", "S&P 500", "D", 1957, "asset"),
    "WILL5000IND": FredSeries(
        "WILL5000IND", "Wilshire 5000 Total Market Full Cap Index", "D", 1971, "asset"),
    "MORTGAGE30US": FredSeries(
        "MORTGAGE30US", "30-Year Fixed Rate Mortgage Average", "W", 1971, "asset"),
    "FIXHAI": FredSeries(
        "FIXHAI", "Housing Affordability Index (Fixed)", "M", 1986, "asset"),

    # ------------------------------------------------------------------ Energy / commodity (A.6)
    "GASREGW": FredSeries(
        "GASREGW", "U.S. Regular All Formulations Gas Price", "W", 1990, "energy",
        notes="Used in gasoline item of consumption basket."),
    "MCOILWTICO": FredSeries(
        "MCOILWTICO", "Spot Crude Oil Price: WTI", "M", 1986, "energy"),
    "DCOILWTICO": FredSeries(
        "DCOILWTICO", "Crude Oil Prices: WTI (daily)", "D", 1986, "energy"),
    "DHHNGSP": FredSeries(
        "DHHNGSP", "Henry Hub Natural Gas Spot Price", "D", 1997, "energy"),
    "APU000074714": FredSeries(
        "APU000074714", "Avg Price: Electricity per kWh, U.S. City Average", "M", 1978, "energy",
        notes="Used in electricity item of consumption basket."),
    "APU0000703112": FredSeries(
        "APU0000703112", "Avg Price: Ground Beef, 100% Beef, per lb", "M", 1984, "energy",
        notes="Used in ground beef item of consumption basket."),
}


CATEGORIES = ("price", "wage", "productivity", "household", "asset", "energy")


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def get_cache_dir(cache_dir: Path | str | None = None) -> Path:
    """Return the cache directory, creating it if needed."""
    path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path(series_id: str, cache_dir: Path | str | None = None) -> Path:
    """Return the cache file path for a series."""
    return get_cache_dir(cache_dir) / f"{series_id}.csv"


def cache_meta_path(series_id: str, cache_dir: Path | str | None = None) -> Path:
    """Return the cache metadata file path for a series."""
    return get_cache_dir(cache_dir) / f"{series_id}.meta.json"


def is_cached(series_id: str, cache_dir: Path | str | None = None) -> bool:
    """Return True if the series is present in the cache."""
    return cache_path(series_id, cache_dir).is_file()


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------

def get_api_key(api_key: str | None = None) -> str:
    """Resolve the FRED API key from argument or environment variable.

    Raises RuntimeError if no key is available. The check is centralized here
    so individual download functions don't need to repeat the lookup.
    """
    if api_key:
        return api_key
    env_key = os.environ.get("FRED_API_KEY", "").strip()
    if not env_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and either "
            "pass api_key= explicitly or export FRED_API_KEY in your shell."
        )
    return env_key


# ---------------------------------------------------------------------------
# Single-series download
# ---------------------------------------------------------------------------

def download_series(
    series_id: str,
    api_key: str | None = None,
    cache_dir: Path | str | None = None,
    force: bool = False,
    timeout: float = 30.0,
) -> pd.Series:
    """Download a FRED series, cache it, and return it as a pandas Series.

    Parameters
    ----------
    series_id : str
        FRED series ID (e.g. "CPIAUCNS"). Should be in the FRED_SERIES catalog.
    api_key : str | None
        FRED API key. If None, reads from the FRED_API_KEY environment variable.
        Not required if `force` is False and the series is already cached.
    cache_dir : Path | str | None
        Override the default cache directory. None uses data/raw/fred/.
    force : bool
        If True, re-download even if the series is cached.
    timeout : float
        HTTP request timeout in seconds.

    Returns
    -------
    pd.Series
        Indexed by pd.DatetimeIndex (the observation date), values are floats.
        Missing observations (FRED uses ".") are returned as NaN.

    Raises
    ------
    requests.HTTPError
        If the FRED API returns a non-2xx status code.
    RuntimeError
        If api_key is None and FRED_API_KEY is not in the environment AND the
        series is not already cached.
    """
    cpath = cache_path(series_id, cache_dir)
    mpath = cache_meta_path(series_id, cache_dir)

    if cpath.is_file() and not force:
        logger.debug("Loading %s from cache: %s", series_id, cpath)
        return _read_cached_series(cpath)

    key = get_api_key(api_key)
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
    }

    logger.info("Downloading %s from FRED API", series_id)
    response = requests.get(FRED_API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    observations = payload.get("observations", [])
    if not observations:
        raise ValueError(f"FRED API returned no observations for {series_id}")

    series = _parse_observations(observations)
    series.name = series_id

    # Persist
    cpath.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(cpath, header=True)

    meta = {
        "series_id": series_id,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "realtime_start": payload.get("realtime_start"),
        "realtime_end": payload.get("realtime_end"),
        "observation_start": payload.get("observation_start"),
        "observation_end": payload.get("observation_end"),
        "count": payload.get("count", len(observations)),
        "first_obs_date": str(series.index.min().date()),
        "last_obs_date": str(series.index.max().date()),
        "n_observations": len(series),
        "n_missing": int(series.isna().sum()),
        "sha256": _sha256_file(cpath),
    }
    mpath.write_text(json.dumps(meta, indent=2))

    return series


def _parse_observations(observations: list[dict[str, Any]]) -> pd.Series:
    """Parse the FRED API observations array into a pandas Series."""
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for obs in observations:
        date_str = obs.get("date")
        value_str = obs.get("value", "")
        if date_str is None:
            continue
        dates.append(pd.Timestamp(date_str))
        if value_str in ("", "."):
            values.append(float("nan"))
        else:
            try:
                values.append(float(value_str))
            except ValueError:
                values.append(float("nan"))
    return pd.Series(values, index=pd.DatetimeIndex(dates), dtype="float64")


def _read_cached_series(path: Path) -> pd.Series:
    """Read a cached CSV back into a pandas Series."""
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    if df.shape[1] == 0:
        return pd.Series(dtype="float64")
    series = df.iloc[:, 0]
    series.index = pd.DatetimeIndex(series.index)
    return series.astype("float64")


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------

def load_series(
    series_id: str,
    cache_dir: Path | str | None = None,
) -> pd.Series:
    """Load a previously-cached FRED series. Raises FileNotFoundError if not cached."""
    cpath = cache_path(series_id, cache_dir)
    if not cpath.is_file():
        raise FileNotFoundError(
            f"{series_id} is not cached at {cpath}. "
            f"Run download_series('{series_id}') or `make data` first."
        )
    return _read_cached_series(cpath)


def load_meta(
    series_id: str,
    cache_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Load the cache metadata for a series."""
    mpath = cache_meta_path(series_id, cache_dir)
    if not mpath.is_file():
        raise FileNotFoundError(f"No metadata file at {mpath}")
    return json.loads(mpath.read_text())


# ---------------------------------------------------------------------------
# Batch download
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Result of a batch download."""
    succeeded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.succeeded) + len(self.failed) + len(self.skipped)

    def __str__(self) -> str:
        return (f"BatchResult(total={self.total}, "
                f"ok={len(self.succeeded)}, "
                f"failed={len(self.failed)}, "
                f"skipped={len(self.skipped)})")


def download_all(
    api_key: str | None = None,
    cache_dir: Path | str | None = None,
    force: bool = False,
    series_ids: list[str] | None = None,
    rate_limit_sleep: float = FRED_RATE_LIMIT_SLEEP,
) -> BatchResult:
    """Download every series in the catalog (or a specified subset).

    Parameters
    ----------
    api_key : str | None
        FRED API key. Falls back to FRED_API_KEY env var.
    cache_dir : Path | str | None
        Override the default cache directory.
    force : bool
        If True, re-download even cached series.
    series_ids : list[str] | None
        If provided, only download these series. Otherwise download all in FRED_SERIES.
    rate_limit_sleep : float
        Seconds to sleep between successive API calls (default ~0.5s for 120 req/min).

    Returns
    -------
    BatchResult
    """
    targets = series_ids if series_ids is not None else list(FRED_SERIES.keys())
    result = BatchResult()

    for sid in targets:
        if not force and is_cached(sid, cache_dir):
            logger.debug("Skipping %s (already cached; use force=True to re-download)", sid)
            result.skipped.append(sid)
            continue
        try:
            download_series(sid, api_key=api_key, cache_dir=cache_dir, force=force)
            result.succeeded.append(sid)
            time.sleep(rate_limit_sleep)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", sid, exc)
            result.failed[sid] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest(
    cache_dir: Path | str | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build the audit-trail manifest from cached series metadata.

    Writes data/processed/manifest.json (or to a custom location) and returns
    the manifest dict. The manifest is the reproducibility contract: every
    figure in Stage 2 of the paper traces back to a manifest entry.
    """
    from rpps import __version__

    output_path = (Path(output_path) if output_path
                   else DEFAULT_PROCESSED_DIR / "manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "simulator_version": __version__,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_series_in_catalog": len(FRED_SERIES),
        "series": {},
    }

    for sid in FRED_SERIES:
        try:
            meta = load_meta(sid, cache_dir)
            manifest["series"][sid] = meta
        except FileNotFoundError:
            manifest["series"][sid] = {"status": "uncached"}

    cached = sum(1 for v in manifest["series"].values()
                 if v.get("status") != "uncached")
    manifest["n_series_cached"] = cached

    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("Wrote manifest: %s (%d/%d series cached)",
                output_path, cached, len(FRED_SERIES))
    return manifest


# ---------------------------------------------------------------------------
# Catalog query helpers
# ---------------------------------------------------------------------------

def series_by_category(category: str) -> list[FredSeries]:
    """Return all FredSeries entries in a given category."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r}. Valid: {CATEGORIES}")
    return [s for s in FRED_SERIES.values() if s.category == category]


def catalog_summary() -> pd.DataFrame:
    """Return the catalog as a DataFrame (useful for inspection)."""
    rows = [
        {
            "series_id": s.id,
            "category": s.category,
            "frequency": s.frequency,
            "start_year": s.start_year,
            "description": s.description,
            "notes": s.notes,
        }
        for s in FRED_SERIES.values()
    ]
    return pd.DataFrame(rows).sort_values(["category", "series_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(description="FRED data acquisition")
    parser.add_argument("--download-all", action="store_true",
                        help="Download every series in the catalog")
    parser.add_argument("--download", metavar="SERIES_ID",
                        help="Download a single series")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached")
    parser.add_argument("--catalog", action="store_true",
                        help="Print the catalog summary")
    parser.add_argument("--manifest", action="store_true",
                        help="Build the audit manifest")
    parser.add_argument("--cache-dir", default=None,
                        help="Override the cache directory")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(args.verbose, 2)]
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    if args.catalog:
        print(catalog_summary().to_string(index=False))
        return

    if args.download:
        s = download_series(args.download, cache_dir=args.cache_dir, force=args.force)
        print(f"{args.download}: {len(s)} obs, {s.index.min().date()} → {s.index.max().date()}")
        return

    if args.download_all:
        result = download_all(cache_dir=args.cache_dir, force=args.force)
        print(result)
        if result.failed:
            print("Failed series:")
            for sid, err in result.failed.items():
                print(f"  {sid}: {err}")

    if args.manifest:
        m = build_manifest(cache_dir=args.cache_dir)
        print(f"Manifest written. {m['n_series_cached']}/{m['n_series_in_catalog']} cached.")


if __name__ == "__main__":
    _main()
