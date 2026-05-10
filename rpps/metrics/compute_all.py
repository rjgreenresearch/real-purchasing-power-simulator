"""
rpps.metrics.compute_all — batch computation of all three derived metrics.

Orchestration layer for Phase 2. Loads the post-`make data` spliced dataset,
loads the basket panel, computes RPPH / WICR / PRWDI, and writes results to
``data/processed/``.

Designed to be called from the command line:

    python -m rpps.metrics.compute_all --output data/processed

or from Python:

    from rpps.metrics.compute_all import run
    summary = run(output_dir="data/processed")
    print(summary)

Outputs (per metric)
--------------------
    {prefix}_composite.csv     — RPPH only
    {prefix}_by_item.csv       — RPPH only
    {prefix}_panel.csv         — WICR / PRWDI (multi-column)
    {prefix}_audit.json        — every metric

Error handling
--------------
Each metric is computed independently. Failures (missing inputs, insufficient
overlap, etc.) are caught, logged, and recorded in the run summary; one
failing metric does not abort the others.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rpps import __version__ as RPPS_VERSION
from rpps.basket import basket_cost_panel
from rpps.fred_loader import load_series
from rpps.metrics.prwdi import compute_prwdi, save_prwdi_result
from rpps.metrics.rpph import compute_rpph, save_rpph_result
from rpps.metrics.wicr import compute_wicr, save_wicr_result
from rpps.nber_splice import load_spliced_wages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-metric runners (each isolated for independent failure handling)
# ---------------------------------------------------------------------------

def _run_rpph(output_dir: Path, frequency: str) -> dict[str, Any]:
    """Compute RPPH and persist. Returns a status dict."""
    try:
        wages = load_spliced_wages()
        panel = basket_cost_panel(frequency=frequency)
        result = compute_rpph(panel, wages, wage_series_id="AHEMAN+M08142USM055NNBR_spliced")
        paths = save_rpph_result(result, output_dir)
        return {
            "metric": "RPPH",
            "status": "ok",
            "n_observations": result.n_observations,
            "coverage_start": str(result.coverage_start.date()) if result.coverage_start else None,
            "coverage_end": str(result.coverage_end.date()) if result.coverage_end else None,
            "items_used": result.items_used,
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    except Exception as exc:  # noqa: BLE001 — orchestrator boundary
        logger.exception("RPPH computation failed")
        return {"metric": "RPPH", "status": "error", "error": str(exc)}


def _run_wicr(output_dir: Path) -> dict[str, Any]:
    """Compute WICR and persist. Returns a status dict."""
    try:
        wages = load_spliced_wages()
        cpi = load_series("CPIAUCNS")
        result = compute_wicr(wages, cpi)
        paths = save_wicr_result(result, output_dir)
        return {
            "metric": "WICR",
            "status": "ok",
            "n_observations": result.n_observations,
            "n_high_wicr_run_periods": int(result.high_wicr_runs.sum()),
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("WICR computation failed")
        return {"metric": "WICR", "status": "error", "error": str(exc)}


def _run_prwdi(output_dir: Path, base_year: int) -> dict[str, Any]:
    """Compute PRWDI and persist. Returns a status dict."""
    try:
        productivity = load_series("OPHNFB")
        real_comp = load_series("COMPRNFB")
        result = compute_prwdi(
            productivity, real_comp,
            base_year=base_year,
            productivity_id="OPHNFB",
            compensation_id="COMPRNFB",
        )
        paths = save_prwdi_result(result, output_dir)
        return {
            "metric": "PRWDI",
            "status": "ok",
            "base_year": base_year,
            "n_observations": result.n_observations,
            "coverage_start": str(result.coverage_start.date()) if result.coverage_start else None,
            "coverage_end": str(result.coverage_end.date()) if result.coverage_end else None,
            "prwdi_at_end": float(result.prwdi.dropna().iloc[-1]) if result.n_observations else None,
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("PRWDI computation failed")
        return {"metric": "PRWDI", "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run(
    output_dir: str | Path = "data/processed",
    *,
    frequency: str = "M",
    base_year: int = 1947,
) -> dict[str, Any]:
    """Run all three metric computations and write a run-summary JSON.

    Parameters
    ----------
    output_dir : str | Path
        Directory for processed outputs. Created if absent.
    frequency : str, default "M"
        Frequency for the RPPH basket panel ("M", "Q", or "A").
    base_year : int, default 1947
        Base year for PRWDI normalization.

    Returns
    -------
    dict
        Run summary with per-metric status and the run-summary file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "rpps_version": RPPS_VERSION,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out.resolve()),
        "frequency": frequency,
        "base_year": base_year,
        "results": [],
    }

    summary["results"].append(_run_rpph(out, frequency))
    summary["results"].append(_run_wicr(out))
    summary["results"].append(_run_prwdi(out, base_year))

    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["n_ok"] = sum(1 for r in summary["results"] if r["status"] == "ok")
    summary["n_error"] = sum(1 for r in summary["results"] if r["status"] != "ok")

    summary_path = out / "metrics_run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    summary["run_summary_path"] = str(summary_path)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(summary: dict[str, Any]) -> None:
    print("=" * 72)
    print("rpps.metrics.compute_all - Phase 2 batch run")
    print("=" * 72)
    print(f"Output directory:  {summary['output_dir']}")
    print(f"Frequency:         {summary['frequency']}")
    print(f"PRWDI base year:   {summary['base_year']}")
    print()
    for r in summary["results"]:
        status = "OK   " if r["status"] == "ok" else "ERROR"
        print(f"  [{status}]  {r['metric']:6s}  ", end="")
        if r["status"] == "ok":
            obs = r.get("n_observations", "?")
            cov_a = r.get("coverage_start", "")
            cov_b = r.get("coverage_end", "")
            extra = f"{obs} obs"
            if cov_a and cov_b:
                extra += f"  ({cov_a} to {cov_b})"
            print(extra)
        else:
            print(r.get("error", "(no message)"))
    print()
    print(f"Run summary:       {summary.get('run_summary_path')}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rpps.metrics.compute_all",
        description="Compute RPPH, WICR, and PRWDI from the spliced dataset.",
    )
    parser.add_argument(
        "--output", default="data/processed",
        help="Output directory for processed metrics (default: data/processed)",
    )
    parser.add_argument(
        "--frequency", default="M", choices=["M", "Q", "A"],
        help="Frequency for the RPPH basket panel (default: M)",
    )
    parser.add_argument(
        "--base-year", type=int, default=1947,
        help="PRWDI base year for normalization (default: 1947)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the human-readable summary",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    summary = run(
        output_dir=args.output,
        frequency=args.frequency,
        base_year=args.base_year,
    )

    if not args.quiet:
        _print_summary(summary)

    return 0 if summary["n_error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
