# TESTING.md

Test procedure for the Real Purchasing Power Simulator (`rpps`).

This document is the operational playbook for running tests, diagnosing
failures, fixing bugs, and adding new tests. It complements `README.md`
(user-facing docs) and `SPECIFICATION.md` (paper-section ↔ module mapping).

---

## 1. Quick start

```bash
# From the repo root, after `make install-dev`:
make test              # full suite (offline, deterministic, ~12 s)
make test-cov          # full suite with line-by-line coverage report
make lint              # ruff style and lint
make typecheck         # mypy static type checks
make all               # install-dev + lint + typecheck + test
```

Direct pytest is also available for finer control:

```bash
python -m pytest tests/                              # all tests
python -m pytest tests/test_counterfactual.py        # one file
python -m pytest tests/test_counterfactual.py -v     # verbose, one line per test
python -m pytest -k "bootstrap"                      # name filter
python -m pytest --tb=short                          # short tracebacks (recommended on failure)
python -m pytest --tb=long --pdb                     # drop into debugger on first failure
python -m pytest -x                                  # stop at first failure
python -m pytest --lf                                # rerun only last-failing tests
python -m pytest --co                                # collect-only (list tests without running)
```

The bare `pytest` command is also fine **if** the install location's
`Scripts/` (Windows) or `bin/` (Unix) is on PATH. If `pytest` isn't found
even after `pip install pytest` succeeds, use the `python -m pytest` form
above — it always works because Python finds its own modules without
needing PATH to be set up. The Makefile uses `python -m` invocations
internally so `make test` works regardless.

Expected baseline as of v0.3.7: **277 tests, 0 failed, ~23 s wall-clock**.
If your local run shows materially different numbers without an intervening
code change, something is wrong with the environment, not the code — first
suspect is `pip install -e ".[dev]" --break-system-packages` not having been
re-run after a Python upgrade.

---

## 2. Suite layout

```
tests/
├── test_fred_loader.py        # FRED API client, caching, manifest, batch download
├── test_nber_splice.py        # NBER ↔ FRED splice, adjustment factor, continuity
├── test_basket.py             # Six-item basket, item costs, panel construction
├── test_rpph.py               # Real Purchasing Power Hours metric
├── test_wicr.py               # Wage-Inflation Capture Ratio + sustained-run flagging
├── test_prwdi.py              # Productivity-Real-Wage Decoupling Index
├── test_compute_all.py        # Phase 2 orchestration / batch metric runner
├── test_breaks.py             # Bai-Perron break detection + Quandt-Andrews supremum
├── test_regression.py         # OLS with HAC SE + regime-stratified regression
└── test_counterfactual.py     # 1948–1971 reference-window counterfactual + bootstrap
```

One test file per source module. The test file's name maps deterministically
to the module it covers: `rpps/foo.py` → `tests/test_foo.py`. Tests for sub-
package modules (`rpps/metrics/rpph.py`) drop the package prefix in the
filename (`tests/test_rpph.py`).

### 2.1 Within-file organization

Each test file is organized into `TestX` classes that group tests by
**behavior**, not by surface API. The conventional class set is:

| Class name pattern              | What goes there                                          |
|---------------------------------|----------------------------------------------------------|
| `TestConstants`                 | Module-level constants and defaults                      |
| `TestBasic` / `TestComputeX`    | Happy-path; the function does what its docstring says    |
| `TestResultContract`            | Returns the expected dataclass with all attributes       |
| `TestKnownDgpRecovery`          | Recovers known coefficients from synthetic DGP           |
| `TestEdgeCases`                 | Empty inputs, single-row inputs, NaN handling            |
| `TestErrorHandling`             | `ValueError` / `TypeError` on malformed inputs           |
| `TestAuditAndSerialization`     | Audit dict has expected keys; `to_dict()` is JSON-safe   |
| `TestPersistence`               | `save_*_result` round-trip via `tmp_path`                |
| `TestDeterminism`               | Same seed → same result; different seed → different      |
| `TestHelpers`                   | Module-private `_helper_function` correctness            |

Not every file uses every class. New modules should mirror this taxonomy
when applicable so a contributor can find what they need without reading
every test method.

### 2.2 Naming

- File: `test_<module>.py`
- Class: `Test<Behavior>` in PascalCase
- Method: `test_<lowercase_what_it_verifies>`, e.g.
  `test_recovers_unit_beta_from_coupled_dgp`,
  `test_zero_noise_yields_tight_ci`,
  `test_empty_panel_raises`.

Method names are read top-to-bottom in failure output. They should describe
the **claim being tested**, not the operation. `test_compute_rpph` is wrong;
`test_doubling_wage_halves_hours` is right.

---

## 3. Conventions for fixtures and synthetic data

### 3.1 Determinism is non-negotiable

Every test that uses random numbers **must** seed its RNG explicitly. The
project standard is `np.random.default_rng(seed)` with an integer seed
chosen at the test author's discretion. Do not use `np.random.seed()` (it
sets the global RNG and leaks state between tests).

```python
# CORRECT
rng = np.random.default_rng(42)
x = rng.normal(0.0, 1.0, 200)

# WRONG — no seed → flaky test
x = np.random.normal(0.0, 1.0, 200)

# WRONG — global state, not isolated
np.random.seed(42)
x = np.random.normal(0.0, 1.0, 200)
```

If a test fails intermittently, the first thing to check is RNG seeding.

### 3.2 DGP-builder helpers

Every test file that needs synthetic time-series data defines a private
helper, prefixed with `_`, that takes named parameters governing the data-
generating process. Examples already in the suite:

- `tests/test_counterfactual.py::_build_dgp(start_year, end_year, pre_alpha,
  pre_beta, post_alpha, post_beta, pivot_year, prod_q_growth, prod_q_shock_sd,
  noise_sd, seed)` — productivity-and-compensation pair with a regime change.
- `tests/test_wicr.py::_build_constant_growth_series(start, n_months,
  annual_growth, initial)` — series with constant compound annual growth.
- `tests/test_prwdi.py::_build_quarterly_series(start_year, n_years,
  annual_growth, initial)` — quarterly index series.
- `tests/test_counterfactual.py::_quarterly_index(start_year, end_year)` —
  quarter-end DatetimeIndex spanning a year range.

When adding a new module, the test file gets its own DGP helpers. Reuse
across files via shared `tests/conftest.py` is allowed but not required;
prefer copy-paste of small helpers over coupling test files together.

### 3.3 Fixtures vs. helper functions

- Use `@pytest.fixture` when the same data is reused across many tests in a
  single class or file (keeps it cached within scope).
- Use a `_helper_function` when the test itself wants to pass non-default
  parameters (e.g. specific α, β, regime structure).

Do not parameterize fixtures across regime / parameter sets; use
`@pytest.mark.parametrize` on the test method instead.

### 3.4 Identification: a checklist

Several DGP-recovery tests are sensitive to whether the synthetic data
**identifies** the parameters being recovered. The most common failure mode
is constant regressors (variance zero in `X` → β unidentified).

Before writing a "should recover known coefficients" test, verify:

1. The regressor varies across observations (`X.var() > 0`).
2. The sample size is large enough that OLS / MLE convergence is fast
   (≥50 obs for one regressor; more for several).
3. Noise is small relative to the parameter scale (`noise_sd << |β · σ_x|`).

If recovery fails with `noise_sd=0`, the DGP is rank-deficient. Fix the
DGP, not the test tolerance.

### 3.5 Floating-point comparisons

| Type                      | Idiom                                                    |
|---------------------------|----------------------------------------------------------|
| Scalar                    | `assert x == pytest.approx(target, abs=1e-6)`            |
| Numpy array / Series      | `assert np.allclose(arr.values, target, atol=1e-6)`      |
| Two pandas Series         | `pd.testing.assert_series_equal(a, b, atol=1e-12)`       |
| Two pandas DataFrames     | `pd.testing.assert_frame_equal(a, b, atol=1e-12)`        |

Do **not** write `(series == pytest.approx(scalar)).all()`. It does not do
what you expect; pytest's `approx` does not broadcast over a Series, and
the chained `.all()` collapses to a single comparison that may silently
return `False` even when the values are correct. (This bug appeared in
two test files in v0.1 and was caught only by running the suite — see
section 6.)

### 3.6 Persistence round-trips

Every `save_*_result` function has a corresponding test that:

1. Runs the computation
2. Saves to `tmp_path` (the pytest-provided fixture)
3. Asserts each expected file exists
4. Re-reads the saved CSV with `pd.read_csv(path, index_col=0, parse_dates=True)`
5. Re-reads the saved JSON with `json.load`
6. Asserts at least one column / key is present

Use the `tmp_path` fixture, not hard-coded paths. Pytest cleans `tmp_path`
between tests automatically.

---

## 4. Reading and diagnosing failures

### 4.1 Standard failure anatomy

```
FAILED tests/test_counterfactual.py::TestReferenceFit::test_recovers_unit_beta_from_coupled_dgp
tests/test_counterfactual.py:189: in test_recovers_unit_beta_from_coupled_dgp
    assert result.reference_beta == pytest.approx(1.0, abs=1e-6)
E   assert 2.4999375015624666e-05 == 1.0 ± 1.0e-06
```

Read three things in this order:

1. **The test name** tells you the claim. ("Should recover unit β.")
2. **The line number and assertion** tell you which claim broke.
3. **The Obtained / Expected** tells you what the actual value was.

Then ask:

- Is the test claim correct, or did the spec change?
  → If spec changed, update the test.
- Is the obtained value plausible, given the DGP?
  → If `obtained ≈ 0` when β=1 was expected, the DGP probably has zero
    variance in the regressor. (See section 3.4.)
- Is there a numerical-precision issue?
  → Loosen `abs=` tolerance only if the deviation has a known cause
    (compounding over 50+ years, bootstrap variance, etc.). Otherwise
    investigate.

### 4.2 Common failure patterns and what they mean

| Symptom                                                 | Likely cause                                               |
|---------------------------------------------------------|------------------------------------------------------------|
| `ValueError: regex pattern did not match`               | Error message changed; update the `match=` regex           |
| Coefficient recovery fails at `noise_sd=0`              | Rank-deficient DGP; regressor has zero variance            |
| Bootstrap CI test passes locally, fails in CI           | Different `numpy` version, different default RNG bit-stream|
| `pytest.approx` against Series returns False everywhere | Use `np.allclose(arr.values, target)` instead              |
| Tests pass individually but fail when run together      | Global state leak (`np.random.seed`, env vars, file I/O)   |
| `TypeError: data.index must be a DatetimeIndex`         | Test passed a plain RangeIndex; wrap in `pd.DatetimeIndex` |
| `AssertionError: insufficient overlap`                  | Test fixtures don't share enough common dates              |
| Coverage drops on a module                              | New code path added without a corresponding test           |

### 4.3 When a test is correct but the code is broken

That's a real bug. Three-step protocol:

1. **Reproduce in isolation**: `pytest tests/test_X.py::TestY::test_z -v`.
   Confirm the failure is deterministic. If it isn't, fix the seeding
   (section 3.1) before doing anything else.
2. **Add a regression test if missing**: if the bug isn't covered by an
   existing assertion, write a test that fails *before* the fix and
   passes *after*. Bug fixes that don't ship with a regression test are
   rejected.
3. **Fix and verify**: make the change in `rpps/`, re-run the failing
   test, then run the **full suite** (`make test`). Fixes for one bug
   that break unrelated tests get caught here.

### 4.4 When a test is wrong

Less common, but happens. Symptoms: assertion encodes an obsolete spec; or
the test is "right by coincidence" (the synthetic data happens to satisfy
a property that isn't actually guaranteed by the implementation).

In this case:

1. Write down explicitly what claim *should* be tested.
2. Compare to what the test actually asserts.
3. Rewrite the test to match the corrected claim.
4. Add a comment in the test explaining the prior incorrect form so the
   git history is searchable for the bug class.

---

## 5. Adding tests for new code

### 5.1 New module

When you add `rpps/foo.py`:

1. Create `tests/test_foo.py` with the section 2.1 class taxonomy.
2. At minimum: one `TestBasic` test, one `TestEdgeCases` test for empty/
   single-row input, one `TestErrorHandling` test per documented `raise`,
   one `TestPersistence` round-trip if the module has `save_*_result`,
   one `TestAuditAndSerialization` test verifying audit keys.
3. Run `pytest tests/test_foo.py -v` until green.
4. Run `make test` to confirm no regressions.
5. Run `make test-cov` and verify the new module is at ≥85% line coverage.

### 5.2 Bug fix

The convention is **regression-test-first**:

```bash
# 1. Write the failing test (or modify an existing one to fail).
pytest tests/test_foo.py::TestBugfix::test_specific_bug -v
# Expect: FAILED with the bug's symptom

# 2. Fix the code in rpps/foo.py.
# 3. Re-run.
pytest tests/test_foo.py::TestBugfix::test_specific_bug -v
# Expect: PASSED

# 4. Run the full suite to check nothing else broke.
make test

# 5. Commit the test and the fix together.
git add rpps/foo.py tests/test_foo.py
git commit -m "fix: <bug summary>; add regression test"
```

### 5.3 New feature in existing module

Add tests to the existing `test_<module>.py` file. If the feature
introduces a new behavioral category that doesn't fit existing
`TestX` classes, add a new class. Avoid mixing unrelated concerns
in one class.

---

## 6. Coverage interpretation

Run `make test-cov` to produce a per-module breakdown. The v0.3.7 baseline:

| Module                        | Coverage | Acceptable? |
|-------------------------------|----------|-------------|
| `rpps/__init__.py`            | 100%     | ✅          |
| `rpps/basket.py`              | 94%      | ✅          |
| `rpps/breaks.py`              | 92%      | ✅          |
| `rpps/counterfactual.py`      | 96%      | ✅          |
| `rpps/fred_loader.py`         | 79%      | ⚠️ Live-FRED HTTP paths uncovered (intentional; see §10.3) |
| `rpps/metrics/__init__.py`    | 100%     | ✅          |
| `rpps/metrics/compute_all.py` | 99%      | ✅          |
| `rpps/metrics/prwdi.py`       | 92%      | ✅          |
| `rpps/metrics/rpph.py`        | 95%      | ✅          |
| `rpps/metrics/wicr.py`        | 94%      | ✅          |
| `rpps/nber_splice.py`         | 99%      | ✅          |
| `rpps/regression.py`          | 95%      | ✅          |
| **Total**                     | **93%**  | ✅          |

**Hard rule**: core computational paths must be ≥85% covered. CLI entry
points and network-error branches are exempt because they require live
FRED API access to exercise meaningfully.

If a PR drops a core module below 85%, either add tests or document why
the new code is intrinsically untestable (rare; usually a code smell).

---

## 7. Local development workflow

### 7.1 Tight loop while editing

```bash
# Run only the file you're working on, with verbose output
pytest tests/test_breaks.py -v

# Run only the test you just modified
pytest tests/test_breaks.py::TestBaiPerronBasic::test_detects_single_break -v

# Watch mode (requires pytest-watch; not in dev deps by default)
ptw -- tests/test_breaks.py
```

### 7.2 Pre-commit checklist

Before any commit that touches `rpps/` or `tests/`:

```bash
make all
```

This runs lint → typecheck → tests in order. Any failure blocks the commit.
If you only changed docs, you can skip `make all` but still run `make lint`
on the off chance a code-block in the README has a typo that ruff catches.

### 7.3 Speeding up the suite during heavy iteration

The full suite runs in ~12 s, which is fast enough that you should
generally just run all of it. If you must iterate faster:

```bash
# Skip slow tests (mark them with @pytest.mark.slow and configure pyproject)
pytest -m "not slow"

# Drop bootstrap tests temporarily (they dominate counterfactual runtime)
pytest -k "not bootstrap and not h4_plausibility"
```

But always run the full suite once before opening a PR.

---

## 8. CI-equivalent local commands

The repo doesn't ship with a GitHub Actions workflow yet. The local
equivalent is:

```bash
make all
```

which is the contract a CI workflow should reproduce. When CI is added,
its single-source-of-truth assertion should be: `make all` exits 0.

---

## 9. Platform notes

The suite is tested on Linux (Ubuntu 24, Python 3.10/3.12), macOS, and
Windows 11 (Python 3.11). Three Windows-specific gotchas are worth
flagging:

### 9.1 `pytest` not on PATH

After `pip install pytest` (or `pip install -e ".[dev]"`), the `pytest`
executable is placed under `C:\Python3xx\Scripts\` (or wherever `pip`
puts entry points). Windows installers do not always add this directory
to `PATH`, so `pytest --version` fails even though pytest is installed.

**Fix**: invoke through Python directly:
```cmd
python -m pytest tests
```
This is what the Makefile uses internally. The same applies to `ruff`,
`mypy`, and `pip` itself if Scripts/ isn't on PATH.

To put `Scripts/` on PATH permanently, append it to the user PATH
environment variable via System Properties → Environment Variables, or
use a virtual environment whose activate script does this for you.

### 9.2 Console codec and Unicode characters

The Windows console default codec is `cp1252`, which cannot encode many
Unicode characters that appear naturally in academic text (em-dash `—`,
right arrow `→`, Greek letters `α β λ`, fractions, etc.). When a Python
program prints such a character to stdout via the default encoding, it
crashes with `UnicodeEncodeError: 'charmap' codec can't encode character`.

The `rpps` CLI strings are all ASCII-only as of v0.3.3, but defensive
practice for any new CLI you add is:

1. Use ASCII in `argparse` `description=`, `help=`, and `epilog=` strings.
2. If you must print Unicode, set `PYTHONIOENCODING=utf-8` in the
   environment, or pass `encoding="utf-8"` to subprocess calls that
   capture child output.
3. Subprocess tests in `rpps` already pass `encoding="utf-8",
   errors="replace"` and inject `PYTHONIOENCODING=utf-8` into the
   child's environment. New subprocess tests should follow the same
   pattern.

### 9.3 `make data` and the FRED API key

The `make data` target downloads FRED + NBER data and builds the spliced
dataset. It requires the `FRED_API_KEY` environment variable to be set
(get a free key at <https://fred.stlouisfed.org/docs/api/api_key.html>).

**Setting the key** (any one of these works):

- macOS / Linux / Git Bash:
  ```bash
  export FRED_API_KEY=your_key_here
  make data
  ```
- Windows cmd.exe:
  ```cmd
  set FRED_API_KEY=your_key_here
  make data
  ```
- Windows PowerShell:
  ```powershell
  $env:FRED_API_KEY = "your_key_here"
  make data
  ```

If `FRED_API_KEY` is unset, the FRED loader prints a one-line error and
exits 1, halting `make data` before the splice step. There is no shell
conditional in the Makefile, so this works identically across cmd.exe,
PowerShell, bash, zsh, and MSYS — any environment in which `make` itself
runs.


## 10. Known caveats and open issues

### 10.1 Quandt-Andrews p-values are approximate

`rpps.breaks.quandt_andrews_test` uses a Hansen-1997-style approximation
rather than the full asymptotic distribution from Andrews 1993 Table I.
Tests verify the test detects known breaks at conventional levels with the
expected sign and ordering, but published applications should cross-check
against the R `strucchange` package's exact tables. The approximation is
documented in the module docstring. This is a methodological caveat, not
a coverage gap; replacing it requires implementing Hansen 1997's tabulated
critical values for `(k, π₀)` pairs.

### 10.2 Bootstrap CI tests

The counterfactual bootstrap CI tests use a mix of `n_bootstrap` settings
(50–1000) chosen for runtime versus precision. The previously-flagged
`test_confidence_level_widens_ci` was restructured in v0.3.2 to assert
*monotonic* widening across four confidence levels (50%, 80%, 95%, 99%) at
`n_bootstrap=1000`, which is robust to the percentile-estimate variance
that smaller bootstrap counts would introduce. The remaining low-`n_bootstrap`
tests verify structural properties (zero-noise → tight CI, seed
reproducibility) where the count doesn't materially affect the assertion.

### 10.3 No tests cover the FRED HTTP path live

By design. All FRED-loader tests use mocked `requests.get` responses or
on-disk fixtures (`data/fixtures/CPIAUCNS_sample.json`). The 79% line
coverage on `rpps/fred_loader.py` reflects exclusively HTTP request
plumbing and rate-limit handling that requires a live network round-trip
to exercise. To verify those paths, run `make data` with `FRED_API_KEY`
set; this is a manual acceptance test, not part of the CI suite.

### 10.4 Phase 4 will introduce new test files

The forthcoming `rpps/cli.py` and `rpps/visualization.py` modules will
each get their own `tests/test_cli.py` and `tests/test_visualization.py`.
The visualization tests will assert on figure structure (number of axes,
axis labels, line counts) rather than on rendered pixels, to keep the
suite headless and deterministic.

### 10.5 Closed in earlier versions

For the historical record:

- **v0.3.7 — Wage splice modern leg correction**. The catalog declared
  `AHETPI` (the splice's modern leg) as starting in 1939, but FRED's
  AHETPI series is "Total Private" production-worker wages and only
  begins in **Jan 1964**. The splice's overlap window of 1939Q1-1942Q4
  therefore had zero observations on the modern side, and `make data`
  crashed with `ValueError: No paired observations in overlap window`.
  Switched `WAGE_MODERN_SERIES` from `AHETPI` to `AHEMAN` (Average
  Hourly Earnings of Production and Nonsupervisory Employees,
  **Manufacturing**), which actually starts in Jan 1939 and is
  industry-consistent with the M08142USM055NNBR NBER manufacturing leg.
  AHETPI remains in the catalog with its true start year (1964) as a
  post-1964 broader-industry reference. Updated the splice docstring,
  the FRED catalog metadata, the README's splice-methodology section,
  the `wage_series_id` audit string in `compute_all`, and DATA_ACQUISITION.md.
  Added three new `TestSpliceOverlapCoverage` regression tests that
  assert the splice's overlap window falls within both legs' declared
  coverage — would have caught this at test time.

- **v0.3.6 — Wage-splice series ID correction**. The `WAGE_LEGACY_SERIES`
  constant declared in v0.3.5 was `M0844AUSM052NNBR`, which is not a real
  FRED series ID. FRED returned a 400 Bad Request when `make data` tried
  to download it. The correct NBER pre-1939 manufacturing wage series on
  FRED is `M08142USM055NNBR` ("Average Hourly Earnings, Twenty-Five
  Manufacturing Industries", monthly, Jan 1920 - Jul 1948), which is what
  the wage splice now uses. Updated the constant, the `FRED_SERIES`
  catalog entry, the README, `DATA_ACQUISITION.md`, the test fixture name,
  and the synthetic-data helper. The `TestCatalogIntegrity` tests added in
  v0.3.5 caught the constant ↔ catalog drift but cannot catch this class
  of "ID is in the catalog but doesn't exist on FRED" bug, which requires
  a live API call. See §10.6 for the remaining residual risk.

### 10.6 Live API drift (unaddressed)

Even with `TestCatalogIntegrity` enforcing constant↔catalog consistency,
the offline test suite cannot catch:

- Series IDs that have never been valid on FRED (typo or hallucination
  during catalog editing — what bit us in v0.3.5)
- Series IDs that were valid but have since been discontinued or renamed
  (what bit us with `WILL5000IND` in v0.3.4 → v0.3.5)

Validating these requires live network access and an API key, which we
keep out of the CI suite by design (see §10.3). The pragmatic mitigation
is a manual acceptance check whenever the catalog changes: after editing
`FRED_SERIES`, run `make data` against a real `FRED_API_KEY` once before
merging. If any series 400s, fix the catalog entry.

A `tools/verify_catalog.py` script that hits FRED for each catalog entry
and reports validity is a candidate for v0.4 — useful for a maintainer
adding new series without having to do a full `make data` run.

- **v0.3.5 — FRED catalog integrity**. Added `M0844AUSM052NNBR` (the NBER
  pre-1939 manufacturing wage series) to `FRED_SERIES`. The wage splice
  declared this constant as `WAGE_LEGACY_SERIES` but the catalog never
  included it, so `make data` downloaded 43-of-44 series successfully and
  then `make data`'s splice step crashed with `FileNotFoundError`. Removed
  `WILL5000IND` from the catalog: FRED removed all Wilshire indices on
  2024-06-03, and `SP500` (already in the catalog) provides the equity-
  market signal. Added a new `TestCatalogIntegrity` test class that asserts
  every series referenced by either the splices or `compute_all` is in
  `FRED_SERIES`, so this class of "catalog drift" bug fails at test time
  rather than at `make data` time.

- **v0.3.4 — `make data` portability + FRED CLI UX**. Replaced the
  `[ -z "$$FRED_API_KEY" ]` POSIX-shell precheck in the Makefile with
  delegation to the FRED loader's own startup check, so `make data`
  works under bare cmd.exe (the previous version exited with
  `-z was unexpected at this time`). Added a top-level RuntimeError
  handler in `rpps.fred_loader._main` that prints a clean one-line
  error and exits 1 when `FRED_API_KEY` is missing, instead of dumping
  a Python traceback. Audited and fixed two more Unicode-stdout bugs
  (em-dash in `compute_all._print_summary` header, right-arrow in
  `fred_loader --download` summary line) plus one Unicode-logging bug
  (`±` in `prwdi.compute_prwdi` warning). Added three regression tests
  exercising the missing-API-key CLI paths (`--download-all`,
  `--download SERIES`, and the `--catalog` happy path that doesn't
  need a key).

- **v0.3.3 — Windows portability**. Replaced the Unicode `→` arrow in the
  `nber_splice` argparse description with ASCII `->` (the `→` crashed the
  CLI on Windows console hosts whose default codec is cp1252). Hardened
  both subprocess CLI tests with `encoding="utf-8", errors="replace"` and
  injected `PYTHONIOENCODING=utf-8` into the child environment so the
  tests are portable across host codecs. Rewrote the Makefile to invoke
  Python tools as `python -m <tool>` rather than as bare executables, so
  `make test`, `make lint`, `make typecheck` work on Windows even when
  `C:\Python3xx\Scripts\` isn't on PATH.

- **v0.3.2 — `compute_all` orchestration coverage** (was 75%, now 99%).
  Added tests for `_print_summary` formatting (both ok and error rows),
  the non-quiet `main()` path, the run-summary version-tracking field
  (regression test for a hardcoded-version bug), and the `__main__`
  guard via subprocess.
- **v0.3.2 — `nber_splice` orchestration coverage** (was 58%, now 99%).
  Added tests for `load_kendrick_productivity` and its error paths,
  `_check_boundary_continuity` edge cases, the wage and productivity
  splice builders via monkeypatched FRED, the `build_spliced_dataset`
  orchestrator (including the missing-Kendrick graceful-skip path), and
  the `_main` CLI. Surfaced and fixed a `relative_to(REPO_ROOT)` bug
  that crashed any external-output-directory invocation.
- **v0.3.2 — `wicr.py` edge cases** (was 86%, now 94%). Added tests for
  the empty-CPI branch, non-DatetimeIndex coercion, and the
  `_detect_freq` quarterly/annual/single-observation branches. Surfaced
  and fixed a `pd.infer_freq` crash on indices with fewer than 3 dates.
- **v0.3.2 — `test_confidence_level_widens_ci` flakiness**. Replaced
  the pairwise (50% vs 99%) comparison at `n_bootstrap=200` with a
  monotonic check across four levels at `n_bootstrap=1000`.

---

## 11. Glossary

- **DGP** — Data-Generating Process. The synthetic recipe used by a test
  to construct inputs whose properties are known a priori.
- **HAC SE** — Heteroskedasticity and Autocorrelation Consistent standard
  errors (Newey-West 1987 / Andrews 1991).
- **PELT** — Pruned Exact Linear Time (Killick, Fearnhead, Eckley 2012);
  the change-point algorithm used by `ruptures` for Bai-Perron-style
  break detection.
- **Reference window** — The 1948–1971 period whose productivity-distribution
  coefficients are estimated and projected forward by the counterfactual.
- **Pivot date** — The last observation in the reference window; the
  counterfactual series is pinned to the actual value at this date.
- **Sustained run** — In the WICR threshold test, a sequence of `≥k`
  consecutive periods above the high threshold (default `k=8`).

---

*Last updated: v0.3.7. Maintained by the `rpps` authors.*
