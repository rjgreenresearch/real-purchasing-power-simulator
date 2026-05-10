# Specification

This document maps each module in the Real Purchasing Power Simulator (`rpps`) to a specific section of the companion working paper:

> Green, R. J. (2026). *The Inflationary Yardstick: A Mutual Threshold Saturation Critique of Nominal Wage Growth as a Welfare Metric, 1925–2025.* Working Paper, Stage 1: Theoretical Framework and Pre-Analysis Plan.

The mapping is one-to-one wherever possible. The simulator is the operational implementation of the methodology specified in §§ 4–5 of the paper, designed to test the registered hypotheses in § 6.

---

## Module-to-paper map

| Module | Implements | Paper §§ |
|--------|-----------|----------|
| `rpps.fred_loader` | FRED API client; series catalog | § 4.2, Appendix A |
| `rpps.nber_splice` | Pre-1947 NBER → post-1947 FRED splice with overlap-window level adjustment | § 4.1 |
| `rpps.basket` | Six-item fixed-quantity consumption basket | § 4.3 |
| `rpps.metrics.rpph` | Real Purchasing Power Hours | § 5.1 (RPPH) |
| `rpps.metrics.wicr` | Wage-Inflation Capture Ratio | § 5.1 (WICR) |
| `rpps.metrics.prwdi` | Productivity-Real-Wage Decoupling Index | § 5.1 (PRWDI) |
| `rpps.breaks` | Bai-Perron multiple-structural-break detection | § 5.2 |
| `rpps.regression` | Within-regime regression with HAC standard errors | § 5.3 |
| `rpps.counterfactual` | 1948–1971 productivity-distribution counterfactual | § 5.4 |
| `rpps.visualisation` | Charts and dashboards for paper figures | § 6 (figures) |

---

## Hypothesis-to-test map

The four registered hypotheses in § 6 of the paper map to specific computations in the simulator. Each is registered before estimation; results will be reported in Stage 2 of the paper regardless of outcome.

| Hypothesis | Implementing module | Falsification condition |
|------------|---------------------|------------------------|
| **H1** Regime existence (≥ 2 breaks, BIC + Quandt-Andrews confirmed) | `rpps.breaks` | < 2 breaks detected, or breaks not robust across procedures |
| **H2** Regime-dependent wage-welfare elasticity (β differs across regimes at 5% level) | `rpps.regression` | β statistically constant; single-regime model preferred by IC |
| **H3** WICR threshold structure (β smaller when WICR > 0.80 vs WICR < 0.50) | `rpps.regression` (sub-period split) | Wage-welfare elasticity invariant to WICR level |
| **H4** Counterfactual gap (≥ 20 % cumulative; CIs exclude zero) | `rpps.counterfactual` | Gap statistically indistinguishable from zero |

---

## Reproducibility contract

Every result in Stage 2 of the paper will be reproducible end-to-end from publicly available data via:

```bash
make install
export FRED_API_KEY=your_key
make data        # downloads + splices
make metrics     # computes RPPH, WICR, PRWDI (Phase 2)
make analysis    # runs breaks, regressions, counterfactual (Phase 3)
make figures     # regenerates all paper figures (Phase 4)
```

The cache layer is deterministic: given the same FRED vintage and the same NBER archive snapshot, two runs produce byte-identical processed outputs. FRED revisions are handled by recording the FRED `realtime_start` and `realtime_end` of every download in a manifest committed alongside the processed dataset.

---

## Versioning

Following [SemVer](https://semver.org). The following constitute breaking changes (require major version bump):

- Removal or rename of any FRED series in the catalog
- Change to the splice methodology (overlap window, adjustment formula)
- Change to the basket composition or fixed quantities
- Change to the structural-break procedure (BIC → AIC, trimming parameter)

Non-breaking changes (minor or patch):

- Addition of new robustness specifications
- Addition of new derived metrics that supplement (do not replace) the principal three
- Performance improvements that preserve numerical output
- Documentation expansions

---

## Audit trail

Every computation in the simulator emits structured logging recording:

- Input series IDs, frequencies, vintages
- Splice parameters (overlap window, adjustment factor)
- Specification choices (deflator, wage series, productivity series)
- Random seeds (for bootstrap operations in `counterfactual`)

The audit trail is written to `data/processed/audit.json` and is intended to make every figure in the paper traceable to the exact specification that produced it.
