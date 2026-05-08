# Data Acquisition

This document catalogs every data series used by the Real Purchasing Power Simulator with its source, frequency, coverage, and processing notes. All series are publicly available U.S. government data. No paywalled or proprietary inputs.

---

## 1. FRED series

All FRED series are downloaded via the FRED API:

```
https://api.stlouisfed.org/fred/series/observations
```

A free API key is required: https://fred.stlouisfed.org/docs/api/api_key.html

The simulator reads the key from the `FRED_API_KEY` environment variable. Downloads are cached in `data/raw/fred/` keyed by series ID. The cache is invalidated by the `--force` flag or by deleting the cache directory.

### 1.1 Price series

| Series ID | Description | Frequency | Coverage |
|-----------|-------------|-----------|----------|
| `CPIAUCNS` | CPI for All Urban Consumers: All Items, NSA | Monthly | 1913+ |
| `CPIAUCSL` | CPI for All Urban Consumers: All Items, SA | Monthly | 1947+ |
| `PPIACO` | Producer Price Index by Commodity: All Commodities | Monthly | 1913+ |
| `GDPDEF` | GDP Implicit Price Deflator | Quarterly | 1947+ |
| `PCEPI` | PCE Chain-Type Price Index | Monthly | 1959+ |
| `WPS0561` | PPI by Commodity: Crude Petroleum (Domestic Production) | Monthly | 1913+ |
| `WPU0911` | PPI by Commodity: Pulp, Paper, and Allied Products | Monthly | 1947+ |
| `WPU051` | PPI by Commodity: Coal | Monthly | 1958+ |
| `WPU101` | PPI by Commodity: Iron and Steel | Monthly | 1926+ |

### 1.2 Wage and compensation series

| Series ID | Description | Frequency | Coverage |
|-----------|-------------|-----------|----------|
| `AHETPI` | Avg Hourly Earnings of Production and Nonsupervisory Employees | Monthly | 1939+ |
| `CES0500000003` | Avg Hourly Earnings of All Employees: Total Private | Monthly | 2006+ |
| `COMPRNFB` | Real Compensation per Hour, Nonfarm Business | Quarterly | 1947+ |
| `COMPNFB` | Compensation per Hour, Nonfarm Business (nominal) | Quarterly | 1947+ |
| `LES1252881600Q` | Median Usual Weekly Real Earnings | Quarterly | 1979+ |
| `MEHOINUSA672N` | Real Median Household Income | Annual | 1984+ |
| `MEPAINUSA672N` | Real Mean Household Income | Annual | 1984+ |
| `DSPIC96` | Real Disposable Personal Income | Monthly | 1959+ |
| `A229RX0` | Real Disposable Personal Income: Per Capita | Monthly | 1959+ |
| `PRS85006023` | Nonfarm Business Sector: Hours Worked | Quarterly | 1947+ |

### 1.3 Productivity series

| Series ID | Description | Frequency | Coverage |
|-----------|-------------|-----------|----------|
| `OPHNFB` | Nonfarm Business Sector: Real Output Per Hour | Quarterly | 1947+ |
| `MFGOPH` | Manufacturing Sector: Real Output Per Hour | Quarterly | 1987+ |
| `ULCNFB` | Nonfarm Business Sector: Unit Labor Cost | Quarterly | 1947+ |
| `LABSHPUSA156NRUG` | Share of Labor Compensation in GDP | Annual | 1950+ |
| `PRS85006092` | Nonfarm Business Sector: Implicit Price Deflator | Quarterly | 1947+ |

### 1.4 Household balance sheet

| Series ID | Description | Frequency | Coverage |
|-----------|-------------|-----------|----------|
| `PSAVERT` | Personal Saving Rate | Monthly | 1959+ |
| `TDSP` | Household Debt Service Payments / DPI | Quarterly | 1980+ |
| `FODSP` | Financial Obligations Ratio for Households | Quarterly | 1980+ |
| `TOTALSL` | Total Consumer Credit Owned and Securitized | Monthly | 1943+ |
| `REVOLSL` | Revolving Consumer Credit | Monthly | 1968+ |
| `DRALACBN` | Delinquency Rate, All Loans, All Commercial Banks | Quarterly | 1985+ |
| `DRSFRMACBS` | Delinquency Rate, Single-Family Residential Mortgages | Quarterly | 1991+ |

### 1.5 Asset prices

| Series ID | Description | Frequency | Coverage |
|-----------|-------------|-----------|----------|
| `CSUSHPISA` | S&P/Case-Shiller National Home Price Index | Monthly | 1987+ |
| `MSPUS` | Median Sales Price of Houses Sold | Quarterly | 1963+ |
| `ASPUS` | Average Sales Price of Houses Sold | Quarterly | 1963+ |
| `SP500` | S&P 500 | Daily (→ Monthly) | 1957+ |
| `WILL5000IND` | Wilshire 5000 Total Market Full Cap Index | Daily (→ Monthly) | 1971+ |
| `MORTGAGE30US` | 30-Year Fixed Rate Mortgage Average | Weekly (→ Monthly) | 1971+ |
| `FIXHAI` | Housing Affordability Index (Fixed) | Monthly | 1986+ |

### 1.6 Energy and commodity

| Series ID | Description | Frequency | Coverage |
|-----------|-------------|-----------|----------|
| `GASREGW` | U.S. Regular All Formulations Gas Price | Weekly (→ Monthly) | 1990+ |
| `MCOILWTICO` | Spot Crude Oil Price: WTI | Monthly | 1986+ |
| `DCOILWTICO` | Crude Oil Prices: WTI | Daily (→ Monthly) | 1986+ |
| `DHHNGSP` | Henry Hub Natural Gas Spot Price | Daily (→ Monthly) | 1997+ |
| `APU000074714` | Avg Price: Electricity per kWh, U.S. City Average | Monthly | 1978+ |
| `APU0000703112` | Avg Price: Ground Beef, 100% Beef, per lb. | Monthly | 1984+ |

---

## 2. NBER Macrohistory series (pre-1947 splice)

NBER Macrohistory series are accessed through FRED (most are mirrored under their original NBER series IDs) or directly from the NBER archive at https://www.nber.org/research/data/nber-macrohistory-database.

| Series ID | Description | Frequency | Coverage |
|-----------|-------------|-----------|----------|
| `M0844AUSM052NNBR` | Manufacturing Average Hourly Earnings, U.S. (Mitchell, Burns, Beney) | Monthly | 1923–1942 |
| `M04051USM324NNBR` | Wholesale Price Index of All Commodities (BLS, historical) | Monthly | 1913–1969 |
| Kendrick (1961) productivity index | Pre-1947 productivity, annual | Annual | 1869–1957 |
| Gordon (2016) productivity revision | Pre-1947 productivity with quality adjustment | Annual | 1869–2014 |

The Kendrick and Gordon historical productivity series are not on FRED. They are committed in this repository under `data/external/kendrick_productivity.csv` and `data/external/gordon_productivity.csv`, with full provenance documented in the headers of each file.

---

## 3. External (non-FRED) series

### 3.1 Tuition (NCES Digest of Education Statistics)

Source: U.S. Department of Education, National Center for Education Statistics, *Digest of Education Statistics*, Table 330.10 (Average undergraduate tuition, fees, room, and board rates charged for full-time students in degree-granting postsecondary institutions, by control and level of institution).

URL: https://nces.ed.gov/programs/digest/d23/tables/dt23_330.10.asp

Coverage: 1969–present (annual). Pre-1969 tuition is reconstructed from individual-institution Higher Education General Information Survey (HEGIS) records and is acknowledged as less reliable.

Committed file: `data/external/nces_tuition.csv`. Format: `year, public_4yr_in_state_total, public_4yr_in_state_tuition_fees, source`.

### 3.2 Healthcare (KFF Employer Health Benefits Survey)

Source: Kaiser Family Foundation, *Employer Health Benefits Survey*, annual.

URL: https://www.kff.org/health-costs/report/employer-health-benefits-annual-survey/

Coverage: 1999–present (annual). Pre-1999 healthcare-cost analysis relies on Cutler and Meara (2001) and is presented as illustrative rather than systematic.

Committed file: `data/external/kff_healthcare.csv`. Format: `year, family_premium_total, employee_contribution, employer_contribution, source`.

### 3.3 Pre-1990 retail electricity

Source: U.S. Energy Information Administration, *Annual Energy Review*, Table 8.10 (Average retail prices of electricity).

URL: https://www.eia.gov/totalenergy/data/annual/

Coverage: 1960–1989 supplements the FRED `APU000074714` series which begins in 1978.

Committed file: `data/external/eia_electricity_pre1990.csv`. Format: `year, residential_cents_per_kwh, source`.

---

## 4. Manifest

After any successful `make data` run, the simulator writes a manifest to `data/processed/manifest.json` containing:

- The FRED `realtime_start` and `realtime_end` of every downloaded series
- The vintage hash of each NBER series (SHA-256 of the cached CSV)
- The simulator version and the package versions used in the build
- The splice parameters (overlap windows, adjustment factors) computed at this build

The manifest is committed alongside any published Stage 2 figure to ensure reproducibility against vintage drift in the underlying data.
