"""
run_phase3.py - Driver for the Phase 3 analysis layer.

Loads the spliced wage and productivity series produced by `make data`, plus
the FRED CPI and real-compensation series, then runs all three Phase 3 modules
in sequence:

  1. rpps.breaks       - Bai-Perron-style multivariate break detection on the
                         CPI/PPI/productivity/wage YoY-growth panel
  2. rpps.regression   - within-regime OLS with HAC standard errors of
                         delta-log(RPPH^-1) on delta-log(wage), delta-log(CPI),
                         delta-log(productivity), with cross-regime coefficient
                         tests
  3. rpps.counterfactual - 1948-1971 productivity-distribution counterfactual
                         applied to the realized post-1971 productivity path,
                         with bootstrap confidence intervals

Each module's outputs land in `data/processed/` with the filenames the
`rpps.report` module already looks for, so `make report` automatically picks
up the Phase 3 panel after this script runs.

This is the "Phase 4b orchestrator" piece: until the top-level `rpps.cli`
lands, this stand-alone script is the simplest way to produce all Phase 3
outputs in one shot.

Usage
-----
    python run_phase3.py                          # uses defaults
    python run_phase3.py --skip-counterfactual    # skip slow step
    python run_phase3.py --processed-dir custom/  # override I/O dir
    python run_phase3.py --quick                  # faster bootstrap
    python run_phase3.py -v                       # progress logging

Prerequisites
-------------
    make data       (produces data/processed/spliced_wages.csv,
                     data/processed/spliced_productivity.csv)
    make metrics    (produces data/processed/wicr_panel.csv with the CPI YoY
                     used for break detection input)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rpps.breaks import detect_breaks_baiperron, save_break_result
from rpps.counterfactual import compute_counterfactual, save_counterfactual_result
from rpps.fred_loader import load_series
from rpps.regression import fit_by_regime, save_regression_result

logger = logging.getLogger("phase3")


# ---------------------------------------------------------------------------
# Step 1: Break detection
# ---------------------------------------------------------------------------

def step_breaks(
    processed_dir: Path,
    cache_dir: Path | None,
    penalty_scale: float = 1.0,
    cost: str = "rbf",
) -> dict:
    """Build the multivariate panel and run break detection.

    Panel: quarterly YoY-growth rates for CPI, PPI, productivity, and wages,
    1947Q1-present. The 1947 start matches §6.1's H1 specification.

    Parameters
    ----------
    penalty_scale : float, default 1.0
        Multiplier on the default BIC-style penalty `k * log(n)`. The default
        (1.0) is the registered Stage 1 specification per §5.2. Smaller values
        admit more breaks; larger values are more conservative. As a sensitivity
        check, this driver also reports the number of breaks at scales 0.5 and
        0.25; values around 0.25 typically detect the canonical 1972 (Bretton
        Woods) and 1982 (Volcker) regime transitions on this data scale.
    cost : str, default "rbf"
        ruptures cost function. "rbf" is non-parametric (default). "l2" is
        sum-of-squared-deviations from segment means.
    """
    import math

    import ruptures as rpt

    logger.info("[1/3] Building break-detection panel...")

    cpi = load_series("CPIAUCNS", cache_dir=cache_dir)
    ppi = load_series("PPIACO", cache_dir=cache_dir)

    spliced_wages = pd.read_csv(
        processed_dir / "spliced_wages.csv",
        index_col=0, parse_dates=True,
    ).iloc[:, 0]
    spliced_prod = pd.read_csv(
        processed_dir / "spliced_productivity.csv",
        index_col=0, parse_dates=True,
    ).iloc[:, 0]

    def yoy_quarterly(s: pd.Series, name: str) -> pd.Series:
        q = s.resample("QE").mean()
        return q.pct_change(4).rename(name)

    panel = pd.concat([
        yoy_quarterly(cpi, "cpi_yoy"),
        yoy_quarterly(ppi, "ppi_yoy"),
        yoy_quarterly(spliced_prod, "prod_yoy"),
        yoy_quarterly(spliced_wages, "wage_yoy"),
    ], axis=1, sort=False).sort_index()

    panel = panel.loc["1947-01-01":].dropna()
    logger.info("    panel shape: %s, range %s -> %s",
                panel.shape, panel.index.min().date(), panel.index.max().date())

    # Sensitivity scan: report break counts at multiple penalty scales so the
    # user can see whether the registered specification's null result is
    # robust or an artifact of penalty calibration.
    n, k = panel.shape
    base_penalty = float(k * math.log(n))
    arr = panel.to_numpy(dtype=float)
    min_size = max(2, int(0.10 * n))
    logger.info("    penalty sensitivity (k*log(n)=%.2f, cost=%s):",
                base_penalty, cost)
    for scale in (1.0, 0.5, 0.25, 0.10, 0.05):
        algo = rpt.Pelt(model=cost, min_size=min_size).fit(arr)
        bps = algo.predict(pen=scale * base_penalty)
        n_brk = len(bps) - 1
        if n_brk > 0:
            dates_clean = []
            for i in bps[:-1]:
                ts = panel.index[i]
                dates_clean.append(f"{ts.year}-Q{(ts.month - 1) // 3 + 1}")
            logger.info("      scale=%.2f, pen=%.2f: %d breaks at %s",
                        scale, scale * base_penalty, n_brk, dates_clean)
        else:
            logger.info("      scale=%.2f, pen=%.2f: 0 breaks",
                        scale, scale * base_penalty)

    logger.info("[1/3] Running Bai-Perron break detection (PELT, %s cost, "
                "penalty_scale=%.2f)...", cost, penalty_scale)
    t0 = time.time()
    result = detect_breaks_baiperron(
        panel, trim=0.10, max_breaks=5, cost=cost,
        penalty=penalty_scale * base_penalty,
    )
    logger.info("    detected %d breaks -> %d regimes in %.1fs",
                result.n_breaks, result.n_regimes, time.time() - t0)
    if result.break_dates:
        logger.info("    break dates: %s",
                    [d.strftime("%Y-%m") for d in result.break_dates])

    paths = save_break_result(result, processed_dir, prefix="breaks")
    for kind, p in paths.items():
        logger.info("    wrote %s -> %s", kind, p)

    return {
        "result": result,
        "panel": panel,
        "n_breaks": result.n_breaks,
        "n_regimes": result.n_regimes,
        "break_dates": result.break_dates,
    }


# ---------------------------------------------------------------------------
# Step 2: Within-regime regression
# ---------------------------------------------------------------------------

def step_regression(
    processed_dir: Path,
    cache_dir: Path | None,
    regime_assignments: pd.Series,
) -> dict:
    """Estimate the §5.3 within-regime regression with cross-regime tests.

    Specification (§5.3):
        Δlog(RPPH⁻¹)_t = α + β · Δlog(wage)_t + γ · Δlog(CPI)_t
                       + δ · Δlog(productivity)_t + ε_t

    The composite RPPH cannot anchor this regression because it begins only
    in 2000 (healthcare-data-limited). We use the inverse of the housing-RPPH
    series as the principal LHS — it is the longest-dated RPPH item with a
    1963 start, so the regression spans 1963-present, covering the regime
    transitions of interest.
    """
    logger.info("[2/3] Building regression panel (housing RPPH proxy)...")

    rpph_by_item = pd.read_csv(
        processed_dir / "rpph_by_item.csv",
        index_col=0, parse_dates=True,
    )
    if "housing" not in rpph_by_item.columns:
        raise RuntimeError(
            "rpph_by_item.csv has no 'housing' column. Re-run `make metrics`."
        )

    spliced_wages = pd.read_csv(
        processed_dir / "spliced_wages.csv",
        index_col=0, parse_dates=True,
    ).iloc[:, 0]
    # NOTE: We deliberately load OPHNFB directly here rather than reading the
    # spliced productivity CSV. The spliced series carries the Kendrick (1961)
    # historical productivity at *annual* frequency, which is appropriate for
    # the breaks panel (full 1947+ historical coverage) but collapses the
    # regression panel to annual when joined to monthly wages and prices via
    # `.resample("QE").mean().diff(4).dropna()`. The post-1947 regression
    # only needs the OPHNFB modern leg, which is natively quarterly with 312
    # observations from 1947Q1, sufficient to populate all detected regimes.
    prod_q = load_series("OPHNFB", cache_dir=cache_dir)
    cpi = load_series("CPIAUCNS", cache_dir=cache_dir)

    # Assemble at quarterly frequency, take YoY log changes.
    def to_q_yoy(s: pd.Series, name: str) -> pd.Series:
        return np.log(s.resample("QE").mean()).diff(4).rename(name)

    rpph_inv = 1.0 / rpph_by_item["housing"].dropna()

    df = pd.concat([
        to_q_yoy(rpph_inv, "dlog_rpph_inv"),
        to_q_yoy(spliced_wages, "dlog_wage"),
        to_q_yoy(cpi, "dlog_cpi"),
        to_q_yoy(prod_q, "dlog_prod"),
    ], axis=1, sort=False).sort_index().dropna()

    # Align regime assignments to df's quarterly index.
    regimes = regime_assignments.reindex(df.index, method="nearest").astype(int)

    y = df["dlog_rpph_inv"]
    X = df[["dlog_wage", "dlog_cpi", "dlog_prod"]]

    logger.info("    regression sample: n=%d, regimes covered: %s",
                len(y), sorted(regimes.unique().tolist()))

    logger.info("[2/3] Estimating within-regime OLS-HAC...")
    t0 = time.time()
    result = fit_by_regime(
        y, X, regimes,
        target_coefficient="dlog_wage",
        min_regime_n=20,
    )
    n_skipped = result.audit.get("n_regimes_skipped_undersize", 0)
    skipped = result.audit.get("skipped_regimes", [])
    logger.info("    fitted %d regimes (%d skipped) in %.1fs",
                len(result.by_regime), n_skipped, time.time() - t0)
    if skipped:
        logger.info("    skipped regimes (n<20): %s", skipped)

    paths = save_regression_result(result, processed_dir, prefix="regression")
    for kind, p in paths.items():
        logger.info("    wrote %s -> %s", kind, p)

    # Summary line for the operator: print the per-regime β̂ on dlog_wage.
    for regime_id, res in sorted(result.by_regime.items()):
        coefs = res.coefficient_table()
        if "dlog_wage" in coefs.index:
            row = coefs.loc["dlog_wage"]
            logger.info(
                "    regime %d: dlog_wage coef = %.4f (se %.4f, n=%d)",
                regime_id, row["coef"], row["std_err"], res.n_observations,
            )

    return {"result": result, "n_regimes_fitted": len(result.by_regime)}


# ---------------------------------------------------------------------------
# Step 3: Counterfactual
# ---------------------------------------------------------------------------

def step_counterfactual(
    processed_dir: Path,
    cache_dir: Path | None,
    n_bootstrap: int = 1000,
) -> dict:
    """Run the §5.4 counterfactual: 1948-1971 distribution coefficients
    applied to the realized post-1971 productivity path.

    Inputs
    ------
    productivity        : OPHNFB (FRED, 1947+, quarterly)
    real compensation   : COMPRNFB (FRED, 1947+, quarterly)

    Both are post-1947 series; pre-1947 extension via the spliced productivity
    leg is not needed here because the reference window is 1948-1971.
    """
    logger.info("[3/3] Loading productivity (OPHNFB) and real comp (COMPRNFB)...")
    prod = load_series("OPHNFB", cache_dir=cache_dir)
    comp = load_series("COMPRNFB", cache_dir=cache_dir)

    logger.info("    productivity: %s -> %s, n=%d",
                prod.index.min().date(), prod.index.max().date(), len(prod))
    logger.info("    real comp:    %s -> %s, n=%d",
                comp.index.min().date(), comp.index.max().date(), len(comp))

    logger.info("[3/3] Computing counterfactual (n_bootstrap=%d)...", n_bootstrap)
    t0 = time.time()
    result = compute_counterfactual(
        productivity=prod,
        real_compensation=comp,
        reference_start=1948,
        reference_end=1971,
        n_bootstrap=n_bootstrap,
        random_seed=42,
        confidence_level=0.95,
    )
    logger.info("    final pct gap: %s (95%% CI [%s, %s]) in %.1fs",
                f"{result.final_pct_gap:+.1%}",
                f"{result.final_pct_gap_ci[0]:+.1%}",
                f"{result.final_pct_gap_ci[1]:+.1%}",
                time.time() - t0)

    paths = save_counterfactual_result(result, processed_dir, prefix="counterfactual")
    for kind, p in paths.items():
        logger.info("    wrote %s -> %s", kind, p)

    return {"result": result, "final_pct_gap": result.final_pct_gap}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python run_phase3.py",
        description="Run Phase 3 analysis (breaks, regression, counterfactual) "
                    "on data produced by `make data` and `make metrics`.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--processed-dir", default="data/processed",
        help="Directory holding spliced data and target for Phase 3 outputs "
             "(default: data/processed).",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="FRED cache directory override. By default uses the simulator's "
             "data/raw/fred path.",
    )
    parser.add_argument(
        "--skip-breaks", action="store_true",
        help="Skip the break-detection step (uses existing breaks_regimes.csv "
             "if present, otherwise fails).",
    )
    parser.add_argument(
        "--skip-regression", action="store_true",
        help="Skip the within-regime regression step.",
    )
    parser.add_argument(
        "--skip-counterfactual", action="store_true",
        help="Skip the counterfactual step (slow due to bootstrap).",
    )
    parser.add_argument(
        "--break-penalty-scale", type=float, default=1.0, metavar="X",
        help="Multiplier on the default BIC-style penalty (k*log(n)) for "
             "break detection. Default 1.0 = registered specification. "
             "Smaller values admit more breaks; values around 0.25 typically "
             "detect canonical 1972/1982 regimes on the standard panel. The "
             "driver always reports a sensitivity scan at scales 1.0, 0.5, "
             "0.25 regardless of which scale is used for the saved result.",
    )
    parser.add_argument(
        "--break-cost", default="rbf", choices=["rbf", "l2", "l1"],
        help="ruptures cost function for break detection. Default: rbf "
             "(non-parametric). l2 = sum-of-squared-deviations from segment means.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use n_bootstrap=200 for the counterfactual instead of the "
             "1,000 default. Useful for laptop iteration; not for final results.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=1)

    args = parser.parse_args(argv)

    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(args.verbose, 2)]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    processed_dir = Path(args.processed_dir).resolve()
    if not processed_dir.is_dir():
        print(f"ERROR: --processed-dir does not exist: {processed_dir}",
              file=sys.stderr)
        return 1

    required = ["spliced_wages.csv", "spliced_productivity.csv", "rpph_by_item.csv"]
    missing = [f for f in required if not (processed_dir / f).is_file()]
    if missing:
        print(f"ERROR: required input(s) missing in {processed_dir}: {missing}",
              file=sys.stderr)
        print("Run `make data` and `make metrics` first.", file=sys.stderr)
        return 1

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    n_bootstrap = 200 if args.quick else 1000

    overall_start = time.time()
    summary: dict = {}

    # ---- Step 1: Breaks ---------------------------------------------------
    if args.skip_breaks:
        logger.info("[1/3] SKIPPED (using existing breaks_regimes.csv)")
        regimes_csv = processed_dir / "breaks_regimes.csv"
        if not regimes_csv.is_file():
            print(f"ERROR: --skip-breaks given but {regimes_csv} does not exist.",
                  file=sys.stderr)
            return 1
        regime_assignments = pd.read_csv(
            regimes_csv, index_col=0, parse_dates=True,
        ).iloc[:, 0]
    else:
        try:
            br = step_breaks(
                processed_dir, cache_dir,
                penalty_scale=args.break_penalty_scale,
                cost=args.break_cost,
            )
        except Exception as exc:
            logger.exception("Step 1 (breaks) failed: %s", exc)
            return 2
        summary["breaks"] = {
            "n_breaks": br["n_breaks"],
            "n_regimes": br["n_regimes"],
            "break_dates": [d.strftime("%Y-%m") for d in br["break_dates"]],
        }
        regime_assignments = br["result"].regime_assignments

    # ---- Step 2: Regression -----------------------------------------------
    if args.skip_regression:
        logger.info("[2/3] SKIPPED")
    else:
        try:
            reg = step_regression(processed_dir, cache_dir, regime_assignments)
        except Exception as exc:
            logger.exception("Step 2 (regression) failed: %s", exc)
            return 3
        summary["regression"] = {"n_regimes_fitted": reg["n_regimes_fitted"]}

    # ---- Step 3: Counterfactual -------------------------------------------
    if args.skip_counterfactual:
        logger.info("[3/3] SKIPPED")
    else:
        try:
            cf = step_counterfactual(processed_dir, cache_dir, n_bootstrap)
        except Exception as exc:
            logger.exception("Step 3 (counterfactual) failed: %s", exc)
            return 4
        summary["counterfactual"] = {"final_pct_gap": cf["final_pct_gap"]}

    elapsed = time.time() - overall_start
    print()
    print("=" * 72)
    print(f"Phase 3 complete in {elapsed:.1f}s")
    print("=" * 72)
    if "breaks" in summary:
        b = summary["breaks"]
        print(f"  Breaks:         {b['n_breaks']} detected -> {b['n_regimes']} regimes")
        print(f"                  dates: {', '.join(b['break_dates']) or 'none'}")
    if "regression" in summary:
        print(f"  Regression:     {summary['regression']['n_regimes_fitted']} regimes fitted")
    if "counterfactual" in summary:
        gap = summary["counterfactual"]["final_pct_gap"]
        print(f"  Counterfactual: final pct gap = {gap:+.1%}")
    print()
    print(f"Outputs written to: {processed_dir}")
    print("Run `make report` to regenerate the HTML with Phase 3 included.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
