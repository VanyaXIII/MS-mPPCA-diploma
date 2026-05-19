"""Kupiec (unconditional coverage) and Christoffersen (independence) VaR tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2


# ---------------------------------------------------------------------------
# Core breach series
# ---------------------------------------------------------------------------

def breach_series(returns: np.ndarray, var_series: np.ndarray) -> np.ndarray:
    """Return binary array: 1 where return < VaR (breach), 0 otherwise."""
    return (returns < var_series).astype(int)


# ---------------------------------------------------------------------------
# Kupiec unconditional coverage test (Binomial / LR)
# ---------------------------------------------------------------------------

def kupiec_test(
    hits: np.ndarray, alpha: float
) -> dict[str, float | int]:
    """Kupiec (1995) binomial test for unconditional coverage.

    Parameters
    ----------
    hits  : binary breach series
    alpha : nominal VaR level

    Returns
    -------
    dict with keys: pvalue, pvalue_pass (1=pass), ci_95_pass, ci_99_pass,
                    breach_rate
    """
    n = len(hits)
    n_breach = int(hits.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = binomtest(n_breach, n=n, p=alpha, alternative="two-sided")

    ci_95 = result.proportion_ci(confidence_level=0.95)
    ci_99 = result.proportion_ci(confidence_level=0.99)

    return {
        "breach_rate": result.statistic,
        "pvalue": float(result.pvalue),
        "pvalue_pass": int(result.pvalue >= 0.05),
        "ci_95_pass": int(ci_95.low <= alpha <= ci_95.high),
        "ci_99_pass": int(ci_99.low <= alpha <= ci_99.high),
    }


# ---------------------------------------------------------------------------
# Christoffersen independence test
# ---------------------------------------------------------------------------

def _transition_counts(hits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count transitions (N) and estimate transition matrix (P)."""
    N = np.zeros((2, 2))
    for t in range(1, len(hits)):
        N[hits[t - 1], hits[t]] += 1
    P = np.zeros((2, 2))
    for i in range(2):
        row_sum = N[i].sum()
        if row_sum > 0:
            P[i] = N[i] / row_sum
    return N, P


def christoffersen_test(hits: np.ndarray, alpha: float) -> dict[str, float | int]:
    """Christoffersen independence test.

    Returns
    -------
    dict with keys: stat, pass (1=cannot reject independence at 1 %)
    """
    N, P = _transition_counts(hits)
    n_breach = hits.sum()
    T = len(hits)
    pi_hat = n_breach / T

    # LR statistic — independence vs. first-order Markov
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = []
        for i in range(2):
            for j in range(2):
                if N[i, j] > 0 and P[i, j] > 0:
                    terms.append(N[i, j] * (np.log(P[i, j]) - np.log(pi_hat if j == 1 else 1 - pi_hat)))
        log_lr = 2.0 * sum(terms) if terms else np.nan

    if np.isnan(log_lr):
        return {"stat": np.nan, "pass": -1}

    threshold = chi2.ppf(0.99, df=1)
    return {
        "stat": log_lr,
        "pass": int(log_lr <= threshold),
    }


# ---------------------------------------------------------------------------
# Aggregated backtest for a portfolio
# ---------------------------------------------------------------------------

def backtest_portfolio(
    portfolio_returns: np.ndarray,
    var_series: np.ndarray,
    alpha: float,
) -> dict[str, float | int]:
    """Run all tests for a single portfolio and return a result dict."""
    hits = breach_series(portfolio_returns, var_series)
    kupiec = kupiec_test(hits, alpha)
    christ = christoffersen_test(hits, alpha)
    return {
        **kupiec,
        "ind_stat": christ["stat"],
        "ind_pass": christ["pass"],
        "sum_reserve": float(np.abs(var_series.sum())),
        "max_breach": float(np.max(var_series - portfolio_returns)),
    }


# ---------------------------------------------------------------------------
# Aggregate across portfolios -> summary table
# ---------------------------------------------------------------------------

def aggregate_results(
    portfolios: np.ndarray,
    returns_matrix: np.ndarray,
    vars_matrix: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    """Run backtest for every portfolio and return a summary DataFrame.

    Parameters
    ----------
    portfolios     : (M, D)
    returns_matrix : (T, D) — out-of-sample returns
    vars_matrix    : (M, T) — VaR series per portfolio
    alpha          : VaR level

    Returns
    -------
    DataFrame with columns: breach_rate, pvalue_pass, ci_95_pass, ci_99_pass,
                             ind_pass, sum_reserve, max_breach
    """
    records = []
    for i, weights in enumerate(portfolios):
        port_ret = returns_matrix @ weights
        record = backtest_portfolio(port_ret, vars_matrix[i], alpha)
        records.append(record)

    df = pd.DataFrame(records)
    summary = df[[
        "breach_rate", "pvalue", "pvalue_pass", "ci_95_pass", "ci_99_pass",
        "ind_pass", "sum_reserve", "max_breach",
    ]].mean()

    # For pass columns report as proportion of portfolios passing
    for col in ["pvalue_pass", "ci_95_pass", "ci_99_pass", "ind_pass"]:
        summary[col] = df[col][df[col] >= 0].mean()  # exclude -1 (N/A)

    return summary.to_frame(name=f"alpha={alpha}").T
