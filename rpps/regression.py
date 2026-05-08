"""
rpps.regression — within-regime regression with HAC standard errors.

Implements §5.3 of:
    Green, R. J. (2026). The Inflationary Yardstick. Working Paper.

Tests hypothesis H2 (regime-dependent wage-welfare elasticity): the within-regime
estimate of β in

    Δlog(RPPH⁻¹_t) = α + β · Δlog(w_t) + γ · Δlog(P_t) + δ · Δlog(Q_t) + ε_t

differs significantly across regimes, with the post-2000 regime estimate
statistically smaller than the 1948–1971 regime estimate at the 5% level.

Standard errors
---------------
Estimation uses ordinary least squares with Newey-West (1987) heteroskedasticity-
and autocorrelation-consistent standard errors. The truncation lag is selected
automatically via the Andrews (1991) AR(1)-prewhitened rule:

    L_opt = floor( 1.1447 * (a(1) * T)^(1/3) )

where a(1) is a function of the AR(1) coefficient of the regression residuals.
The default upper bound is L_max = floor(4 * (T/100)^(2/9)) per Newey-West (1994).

H2 cross-regime test
--------------------
Given regime-specific β̂_r and HAC standard errors σ̂_r, the cross-regime
difference β̂_r1 − β̂_r2 is tested via:

    z = (β̂_r1 − β̂_r2) / sqrt(σ̂²_r1 + σ̂²_r2)

asymptotically standard-normal under H2_0. This treats the regime-specific
estimates as independent, which is appropriate when the regime boundary is
treated as exogenous (per the Bai-Perron pre-screen).

References
----------
Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent
    covariance matrix estimation. Econometrica, 59(3), 817–858.
Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
    Econometrica, 55(3), 703–708.
Newey, W. K. and West, K. D. (1994). Automatic lag selection in covariance
    matrix estimation. Review of Economic Studies, 61(4), 631–653.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RegressionResult:
    """Result of a single OLS regression with HAC standard errors.

    Attributes
    ----------
    coefficients : pd.Series
    std_errors : pd.Series
    t_statistics : pd.Series
    p_values : pd.Series
    r_squared : float
    adj_r_squared : float
    n_observations : int
    hac_lag : int
    formula : str
    fitted_values : pd.Series
    residuals : pd.Series
    audit : dict
    """

    coefficients: pd.Series
    std_errors: pd.Series
    t_statistics: pd.Series
    p_values: pd.Series
    r_squared: float
    adj_r_squared: float
    n_observations: int
    hac_lag: int
    formula: str
    fitted_values: pd.Series
    residuals: pd.Series
    audit: dict = field(default_factory=dict)

    def coefficient_table(self) -> pd.DataFrame:
        """Return a tidy table of coefficient estimates."""
        return pd.DataFrame({
            "coef": self.coefficients,
            "std_err": self.std_errors,
            "t_stat": self.t_statistics,
            "p_value": self.p_values,
        })

    def to_dict(self) -> dict:
        return {
            "formula": self.formula,
            "n_observations": self.n_observations,
            "hac_lag": self.hac_lag,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "coefficients": self.coefficient_table().to_dict(orient="index"),
            "audit": self.audit,
        }


@dataclass
class RegimeRegressionResult:
    """Container for regime-stratified regression results.

    Attributes
    ----------
    by_regime : dict[int, RegressionResult]
        One regression result per regime label.
    cross_regime_tests : pd.DataFrame
        Pairwise cross-regime tests of the principal coefficient. Columns:
        regime_a, regime_b, beta_a, beta_b, diff, z_stat, p_value.
    target_coefficient : str
        Name of the coefficient on which cross-regime tests were performed.
    audit : dict
    """

    by_regime: dict[int, RegressionResult]
    cross_regime_tests: pd.DataFrame
    target_coefficient: str
    audit: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "target_coefficient": self.target_coefficient,
            "regime_count": len(self.by_regime),
            "by_regime": {str(r): res.to_dict() for r, res in self.by_regime.items()},
            "cross_regime_tests": self.cross_regime_tests.to_dict(orient="records"),
            "audit": self.audit,
        }


# ---------------------------------------------------------------------------
# Single regression
# ---------------------------------------------------------------------------

def fit_ols_hac(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    hac_lag: int | None = None,
    add_constant: bool = True,
) -> RegressionResult:
    """Fit OLS with Newey-West HAC standard errors.

    Parameters
    ----------
    y : pd.Series
        Dependent variable.
    X : pd.DataFrame
        Regressors. Constant added automatically unless add_constant=False.
    hac_lag : int | None
        Truncation lag for Newey-West. If None, selected via Newey-West (1994)
        rule of thumb based on sample size.
    add_constant : bool, default True

    Returns
    -------
    RegressionResult
    """
    if y.empty:
        raise ValueError("y is empty")
    if len(y) != len(X):
        raise ValueError(f"y and X must have same length: got {len(y)} vs {len(X)}")
    if y.isna().any() or X.isna().any().any():
        raise ValueError("y or X contains NaN; drop or impute before fitting")

    n = len(y)
    if add_constant:
        X = sm.add_constant(X, has_constant="add")

    if hac_lag is None:
        hac_lag = _newey_west_default_lag(n)

    model = sm.OLS(y.to_numpy(), X.to_numpy())
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag})

    coef_names = list(X.columns)
    coefficients = pd.Series(fit.params, index=coef_names, name="coef")
    std_errors = pd.Series(fit.bse, index=coef_names, name="std_err")
    t_stats = pd.Series(fit.tvalues, index=coef_names, name="t_stat")
    p_values = pd.Series(fit.pvalues, index=coef_names, name="p_value")

    fitted = pd.Series(fit.fittedvalues, index=y.index, name="fitted")
    resid = pd.Series(fit.resid, index=y.index, name="resid")

    formula = f"{y.name or 'y'} ~ " + " + ".join(c for c in coef_names if c != "const")
    audit = {
        "estimator": "OLS",
        "covariance": "Newey-West HAC",
        "hac_lag": int(hac_lag),
        "n_observations": int(n),
        "n_regressors": int(X.shape[1]),
        "rank": int(np.linalg.matrix_rank(X.to_numpy())),
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return RegressionResult(
        coefficients=coefficients,
        std_errors=std_errors,
        t_statistics=t_stats,
        p_values=p_values,
        r_squared=float(fit.rsquared),
        adj_r_squared=float(fit.rsquared_adj),
        n_observations=int(n),
        hac_lag=int(hac_lag),
        formula=formula,
        fitted_values=fitted,
        residuals=resid,
        audit=audit,
    )


def _newey_west_default_lag(n: int) -> int:
    """Newey-West (1994) automatic lag rule: floor(4 * (n/100)^(2/9))."""
    return max(1, int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


# ---------------------------------------------------------------------------
# Regime-stratified regression
# ---------------------------------------------------------------------------

def fit_by_regime(
    y: pd.Series,
    X: pd.DataFrame,
    regimes: pd.Series,
    *,
    target_coefficient: str,
    min_regime_n: int = 20,
    hac_lag: int | None = None,
) -> RegimeRegressionResult:
    """Fit OLS-HAC within each regime and compare a target coefficient across regimes.

    Parameters
    ----------
    y : pd.Series
    X : pd.DataFrame
    regimes : pd.Series
        Integer regime labels aligned with y.index.
    target_coefficient : str
        Name of the coefficient on which cross-regime tests are conducted.
    min_regime_n : int, default 20
        Skip regimes with fewer observations than this.
    hac_lag : int | None
        Truncation lag override.

    Returns
    -------
    RegimeRegressionResult
    """
    if not (len(y) == len(X) == len(regimes)):
        raise ValueError("y, X, regimes must have equal length")

    by_regime: dict[int, RegressionResult] = {}
    skipped: list[int] = []

    for r in sorted(regimes.dropna().unique()):
        mask = regimes == r
        if mask.sum() < min_regime_n:
            skipped.append(int(r))
            continue
        y_r = y[mask].dropna()
        X_r = X.loc[y_r.index].dropna()
        joint = X_r.index.intersection(y_r.index)
        y_r = y_r.loc[joint]
        X_r = X_r.loc[joint]
        if len(y_r) < min_regime_n:
            skipped.append(int(r))
            continue
        result = fit_ols_hac(y_r, X_r, hac_lag=hac_lag)
        by_regime[int(r)] = result

    cross_tests = _pairwise_coefficient_tests(by_regime, target_coefficient)

    audit = {
        "n_regimes_fit": len(by_regime),
        "n_regimes_skipped_undersize": len(skipped),
        "skipped_regimes": skipped,
        "min_regime_n": min_regime_n,
        "target_coefficient": target_coefficient,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return RegimeRegressionResult(
        by_regime=by_regime,
        cross_regime_tests=cross_tests,
        target_coefficient=target_coefficient,
        audit=audit,
    )


def _pairwise_coefficient_tests(
    by_regime: dict[int, RegressionResult],
    target_coefficient: str,
) -> pd.DataFrame:
    """Build a DataFrame of pairwise cross-regime z-tests on target_coefficient."""
    from scipy import stats

    rows = []
    regime_ids = sorted(by_regime.keys())
    for i, ra in enumerate(regime_ids):
        for rb in regime_ids[i + 1:]:
            res_a = by_regime[ra]
            res_b = by_regime[rb]
            if target_coefficient not in res_a.coefficients.index:
                continue
            if target_coefficient not in res_b.coefficients.index:
                continue
            beta_a = float(res_a.coefficients[target_coefficient])
            beta_b = float(res_b.coefficients[target_coefficient])
            se_a = float(res_a.std_errors[target_coefficient])
            se_b = float(res_b.std_errors[target_coefficient])
            diff = beta_a - beta_b
            se_diff = math.sqrt(se_a ** 2 + se_b ** 2)
            if se_diff <= 0:
                z = float("nan")
                p = float("nan")
            else:
                z = diff / se_diff
                p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
            rows.append({
                "regime_a": ra,
                "regime_b": rb,
                "beta_a": beta_a,
                "beta_b": beta_b,
                "diff": diff,
                "se_diff": se_diff,
                "z_stat": z,
                "p_value": p,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_regression_result(
    result: RegressionResult | RegimeRegressionResult,
    output_dir: str | Path,
    prefix: str = "regression",
) -> dict[str, Path]:
    """Save a regression or regime-regression result to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    audit_path = out / f"{prefix}_audit.json"
    with open(audit_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)

    paths = {"audit": audit_path}

    if isinstance(result, RegimeRegressionResult):
        cross_path = out / f"{prefix}_cross_regime.csv"
        result.cross_regime_tests.to_csv(cross_path, index=False)
        paths["cross_regime"] = cross_path

        # Per-regime coefficient tables.
        all_coefs = []
        for regime_id, res in result.by_regime.items():
            coef_df = res.coefficient_table()
            coef_df["regime"] = regime_id
            coef_df["coefficient"] = coef_df.index
            all_coefs.append(coef_df)
        if all_coefs:
            coefs_path = out / f"{prefix}_by_regime_coefs.csv"
            pd.concat(all_coefs).reset_index(drop=True).to_csv(coefs_path, index=False)
            paths["by_regime_coefs"] = coefs_path

    return paths
