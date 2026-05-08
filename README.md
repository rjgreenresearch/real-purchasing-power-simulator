# Real Purchasing Power Simulator

**MTS Pillar 6 — Welfare Measurement Architecture**

A reproducible Python instrument for testing whether the conventional identification of economic strength with rising nominal wages — and the policy infrastructure that targets nominal aggregates — represents a measurement architecture that systematically masks compound economic fragility under the Mutual Threshold Saturation (MTS) doctrine.

Implements the empirical strategy registered in:

> Green, R. J. (2026). *The Inflationary Yardstick: A Mutual Threshold Saturation Critique of Nominal Wage Growth as a Welfare Metric, 1925–2025.* Working Paper, Stage 1: Theoretical Framework and Pre-Analysis Plan. SSRN (forthcoming).

---

## What this simulator does

1. **Acquires** approximately fifty FRED series and twelve NBER Macrohistory series spanning 1913–2025. All sources are publicly available U.S. government data. No paywalled or proprietary inputs.
2. **Splices** pre-1947 NBER series to post-1947 FRED series using documented overlap-window level adjustments. Produces a continuous 1925–2025 dataset.
3. **Constructs** three derived metrics: Real Purchasing Power Hours (RPPH), the Wage-Inflation Capture Ratio (WICR), and the Productivity-Real-Wage Decoupling Index (PRWDI).
4. **Detects** structural breaks in the joint behavior of CPI, PPI, productivity, and nominal-wage growth using the Bai-Perron procedure.
5. **Estimates** within-regime regressions of real-purchasing-power growth on nominal-wage growth with HAC standard errors.
6. **Computes** the counterfactual "lost welfare" series — what median real compensation would have been if 1948–1971 productivity-distribution dynamics had persisted through 2025.

Each step is reproducible end-to-end from publicly available data with a single `make` command.

---

## Status

Phase 1 (data foundation), Phase 2 (derived metrics), and Phase 3 (analysis layer) are implemented in this release.

| Phase | Module | Status |
|-------|--------|--------|
| 1 | FRED + NBER acquisition, splice, basket definition | ✅ Implemented (v0.1) |
| 2 | RPPH / WICR / PRWDI metric computation | ✅ Implemented (v0.2) |
| 3 | Bai-Perron break detection, within-regime regression, counterfactual | ✅ Implemented (v0.3) |
| 4 | CLI, notebooks, end-to-end reproduction | ⏳ Forthcoming |

---

## Quick start

```bash
# Install
make install

# Run the test suite (offline, uses fixture data)
make test

# Get a free FRED API key:  https://fred.stlouisfed.org/docs/api/api_key.html
export FRED_API_KEY=your_key_here

# Build the spliced 1925–2025 dataset (~2 minutes; ~12 MB cached)
make data

# Inspect the spliced dataset
python -c "from rpps.nber_splice import load_spliced_wages; print(load_spliced_wages().describe())"
```

### Compute the three derived metrics (Phase 2, v0.2+)

```python
import pandas as pd
from rpps.fred_loader import load_series
from rpps.basket import basket_cost_panel
from rpps.metrics import compute_rpph, compute_wicr, compute_prwdi

# RPPH — Real Purchasing Power Hours
panel = basket_cost_panel(frequency="M")               # nominal cost of the fixed basket
wages = load_series("AHETPI")                          # production-worker hourly wage
rpph = compute_rpph(panel, wages, wage_series_id="AHETPI")
print(rpph.composite.tail())                           # hours of labor to buy the basket

# WICR — Wage-Inflation Capture Ratio
cpi = load_series("CPIAUCNS")
wicr = compute_wicr(wages, cpi)
print(wicr.regime_label.value_counts())                # low / medium / high regime tally
print(int(wicr.high_wicr_runs.sum()), "periods in sustained-high-WICR runs")

# PRWDI — Productivity-Real-Wage Decoupling Index
prod = load_series("OPHNFB")
real_comp = load_series("COMPRNFB")
prwdi = compute_prwdi(prod, real_comp, base_year=1947)
print(prwdi.prwdi.tail())                              # cumulative decoupling vs 1947
```

### Run the analysis layer (Phase 3, v0.3+)

```python
from rpps.fred_loader import load_series
from rpps.breaks import detect_breaks_baiperron, quandt_andrews_test
from rpps.regression import fit_ols_hac, fit_by_regime
from rpps.counterfactual import compute_counterfactual

# Structural-break detection (Bai-Perron via PELT)
cpi_growth = load_series("CPIAUCNS").pct_change(12).dropna()
break_result = detect_breaks_baiperron(cpi_growth, max_breaks=5)
print(break_result.regime_assignments.value_counts())

# Within-regime regression with HAC standard errors
ols = fit_ols_hac(y, X)                                # y, X built from your data
print(ols.params, ols.se_hac)

# Counterfactual: 1948–1971 productivity-distribution coefficients
# applied to the realized 1972–2025 productivity path.
prod = load_series("OPHNFB")
real_comp = load_series("COMPRNFB")
cf = compute_counterfactual(prod, real_comp,
                            reference_start=1948, reference_end=1971)
print(f"Final pct gap: {cf.final_pct_gap:.1%}")
print(f"Bootstrap 95% CI: [{cf.final_pct_gap_ci[0]:.1%}, {cf.final_pct_gap_ci[1]:.1%}]")
```

---

## Architecture

```
real-purchasing-power-simulator/
├── README.md
├── LICENSE                         # Apache 2.0
├── CITATION.cff
├── SPECIFICATION.md                # Maps each module to a section of the working paper
├── DATA_ACQUISITION.md             # FRED + NBER series catalog with download instructions
├── pyproject.toml
├── requirements.txt
├── Makefile
│
├── rpps/
│   ├── __init__.py
│   ├── fred_loader.py              # FRED API client with on-disk caching
│   ├── nber_splice.py              # Pre-1947 NBER → post-1947 FRED splice
│   ├── basket.py                   # Six-item fixed-quantity consumption basket
│   ├── metrics/                    # Phase 2 (implemented)
│   │   ├── rpph.py                 # Real Purchasing Power Hours
│   │   ├── wicr.py                 # Wage-Inflation Capture Ratio
│   │   └── prwdi.py                # Productivity-Real-Wage Decoupling Index
│   ├── breaks.py                   # Phase 3: Bai-Perron structural break detection
│   ├── regression.py               # Phase 3: within-regime regression with HAC SE
│   ├── counterfactual.py           # Phase 3: 1948–1971 counterfactual
│   └── visualisation.py            # Phase 4: chart generation
│
├── data/
│   ├── raw/                        # FRED and NBER cached downloads (gitignored)
│   ├── processed/                  # Spliced dataset + derived metrics (gitignored)
│   ├── external/                   # NCES tuition, KFF healthcare (committed)
│   └── fixtures/                   # Deterministic test fixtures (committed)
│
├── tests/
│   ├── test_fred_loader.py
│   ├── test_nber_splice.py
│   ├── test_basket.py
│   ├── test_rpph.py
│   ├── test_wicr.py
│   ├── test_prwdi.py
│   ├── test_compute_all.py
│   ├── test_breaks.py
│   ├── test_regression.py
│   └── test_counterfactual.py
│
└── notebooks/                      # Phase 4 (forthcoming)
```

---

## The splice methodology (the part most likely to need iteration)

For the principal nominal-wage series, the FRED `AHETPI` (Average Hourly Earnings of Production and Nonsupervisory Employees, 1939+) is the canonical post-1939 source. Pre-1939, the NBER Macrohistory series `M0844AUSM052NNBR` (manufacturing average hourly earnings, monthly, 1923–1942) is the closest available substitute.

The simulator splices using a multiplicative level adjustment computed over the 1939Q1–1942Q4 overlap window:

```
λ = geometric_mean(AHETPI_t / M0844_t)  for t ∈ overlap window
spliced_wage_t = M0844_t · λ            for t < 1939
spliced_wage_t = AHETPI_t                for t ≥ 1939
```

The geometric mean is preferred over the arithmetic mean because both series are positive and the relationship is multiplicative rather than additive. The approach is standard for chained price/wage series; see Boskin et al. (1996) §3 and BLS Handbook of Methods Ch. 10 for the related CPI substitution methodology.

A symmetric splice is applied to the productivity series, joining the Kendrick (1961) annual historical productivity index to the FRED `OPHNFB` quarterly series over the 1947–1957 overlap window.

The splice arithmetic is unit-tested. Continuity at the splice boundary, growth-rate sanity over the overlap window, and reversibility under inverse splice are all verified deterministically.

---

## Data sources

All series are publicly available from the Federal Reserve Bank of St. Louis FRED database (fred.stlouisfed.org) and the NBER Macrohistory Database (nber.org/research/data/nber-macrohistory-database). The complete series catalog is in [`DATA_ACQUISITION.md`](DATA_ACQUISITION.md).

| Data | Source | Frequency | Coverage | Authentication |
|------|--------|-----------|----------|----------------|
| ~50 FRED series | FRED | M / Q / A | Various, 1913+ | Free API key |
| ~12 NBER series | NBER Macrohistory | M / A | 1913–1969 (legacy) | None |
| NCES tuition | Dept. of Education | A | 1969+ | None (committed CSV) |
| KFF healthcare | KFF Employer Health Benefits Survey | A | 1999+ | None (committed CSV) |
| Pre-1969 tuition / pre-1990 electricity | Various historical | A | Sparse | Documented per-source |

No data files are bundled in the repository (other than the small NCES/KFF tables and the test fixtures). All data is downloaded directly from the authoritative source on first run and cached locally for subsequent runs.

---

## Citation

If you use the Real Purchasing Power Simulator in research, policy analysis, or publications, please cite the simulator and the companion working paper:

```bibtex
@software{green_rpps_2026,
  author  = {Green, Robert J.},
  title   = {{Real Purchasing Power Simulator}: {MTS} Pillar 6 — Welfare Measurement Analysis},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/rjgreenresearch/real-purchasing-power-simulator},
  license = {Apache-2.0}
}

@unpublished{green_inflationary_yardstick_2026,
  author = {Green, Robert J.},
  title  = {The Inflationary Yardstick: {A} Mutual Threshold Saturation Critique of Nominal Wage Growth as a Welfare Metric, 1925--2025},
  year   = {2026},
  note   = {Working Paper, Stage 1; companion to forthcoming Bloomberg Opinion piece.}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

---

## Research context

The Real Purchasing Power Simulator is Pillar 6 of the Mutual Threshold Saturation (MTS) research programme:

| Pillar | Paper | Domain | Simulator |
|--------|-------|--------|-----------|
| 1 | MTS Supply Chain | Material dependencies | [mts-doctrine-simulator](https://github.com/rjgreenresearch/mts-doctrine-simulator) |
| 2 | HCTS | Human-capital key-person dependencies | [hcts-simulator](https://github.com/rjgreenresearch/hcts-simulator) |
| 3 | Cost Asymmetry | Compound warfare economics | [cost-asymmetry-simulator](https://github.com/rjgreenresearch/cost-asymmetry-simulator) |
| 4 | Compound Economic Fragility | Domestic economic resilience | [economic-fragility-simulator](https://github.com/rjgreenresearch/economic-fragility-simulator) |
| 5 | Aquifer Depletion | Irreversible natural-resource depletion | [aquifer-depletion-simulator](https://github.com/rjgreenresearch/aquifer-depletion-simulator) |
| **6** | **Inflationary Yardstick (this paper)** | **Welfare-measurement architecture** | **real-purchasing-power-simulator** |

The connecting thesis is that compound threshold saturation — the condition in which simultaneous stress across multiple domains exhausts absorptive capacity multiplicatively — is the structural mechanism underlying both economic crises and national security failures. Pillar 6 is the epistemic dimension: the metric architecture that determines whether the other five pillars are visible or invisible to policy.

---

## License

Apache 2.0. See [`LICENSE`](LICENSE) for full terms.
