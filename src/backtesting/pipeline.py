"""End-to-end backtesting pipeline: rolling fit → VaR → Kupiec/Christoffersen."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.backtest import aggregate_results
from src.backtesting.portfolios import generate_diversified, generate_non_diversified
from src.models.rolling import (
    MSRollingResult, RollingResult,
    fit_rolling, fit_rolling_ms, load, load_ms, save, save_ms,
)
from src.models.var import (
    compute_var_multi_level,
    compute_var_multi_level_fixed_nu,
    compute_var_multi_level_student,
)


def run(
    returns: np.ndarray,
    window: int = 350,
    step: int = 1,
    n_components: int = 3,
    n_clusters: int = 2,
    n_portfolios: int = 200,
    alphas: list[float] | None = None,
    artifact_path: str | None = None,
    force_refit: bool = False,
) -> dict[str, pd.DataFrame | RollingResult]:
    """Full pipeline from log-returns to backtest summary tables.

    Parameters
    ----------
    returns       : (T, D) log-return matrix
    window        : rolling window length
    step          : window stride
    n_components  : mPPCA latent dimension q
    n_clusters    : number of mixture components K
    n_portfolios  : portfolios per type (diversified + non-diversified)
    alphas        : VaR levels, default [0.05, 0.01]
    artifact_path : if given, save/load rolling fit to .npz at this path
    force_refit   : ignore cached artifact and refit from scratch

    Returns
    -------
    {
      "diversified":     pd.DataFrame | RollingResult — backtest summary per alpha level,
      "non_diversified": pd.DataFrame | Rolling Result,
    }
    """
    if alphas is None:
        alphas = [0.05, 0.01]

    D = returns.shape[1]

    # ----- Fit or load rolling mPPCA -----
    result: RollingResult
    if artifact_path and Path(artifact_path).exists() and not force_refit:
        print(f"Loading rolling fit from {artifact_path}")
        result = load(artifact_path)
    else:
        result = fit_rolling(
            returns,
            window=window,
            step=step,
            n_components=n_components,
            n_clusters=n_clusters,
        )
        if artifact_path:
            save(result, artifact_path)
            print(f"Saved rolling fit to {artifact_path}")

    T_out = result.means_hist.shape[0]
    # Out-of-sample returns start after the first window
    oos_returns = returns[window : window + T_out]

    # ----- Generate portfolios -----
    portfolios_div = generate_diversified(n_portfolios, D)
    portfolios_nondiv = generate_non_diversified(n_portfolios, D)

    summaries: dict[str, list[pd.DataFrame]] = {
        "diversified": [],
        "non_diversified": [],
    }

    for alpha in alphas:
        # VaR for diversified
        vars_div = compute_var_multi_level(result, portfolios_div, [alpha])[alpha]
        summary_div = aggregate_results(portfolios_div, oos_returns, vars_div, alpha)
        summaries["diversified"].append(summary_div)

        # VaR for non-diversified
        vars_nondiv = compute_var_multi_level(result, portfolios_nondiv, [alpha])[alpha]
        summary_nondiv = aggregate_results(portfolios_nondiv, oos_returns, vars_nondiv, alpha)
        summaries["non_diversified"].append(summary_nondiv)

    return {
        "diversified": pd.concat(summaries["diversified"]),
        "non_diversified": pd.concat(summaries["non_diversified"]),
        "result": result,
    }


def run_ms(
    returns: np.ndarray,
    window: int = 350,
    step: int = 1,
    n_components: int = 3,
    n_clusters: int = 2,
    n_portfolios: int = 200,
    alphas: list[float] | None = None,
    artifact_path: str | None = None,
    force_refit: bool = False,
    sticky_diag: float = 2.0,
    hmm_eps: float = 0.05,
    emission: str = "normal",
) -> dict[str, pd.DataFrame | MSRollingResult]:
    """Same as run() but uses MS-mPPCA (HMM with mPPCA emissions).

    Parameters
    ----------
    emission : "normal" (Gaussian, default) or "student" (Student-t with
               learnable ν per cluster — fixes 1% VaR coverage failure).

    Returns
    -------
    {
      "diversified":     pd.DataFrame,
      "non_diversified": pd.DataFrame,
      "result":          MSRollingResult,
    }
    """
    if alphas is None:
        alphas = [0.05, 0.01]

    D = returns.shape[1]

    result: MSRollingResult
    if artifact_path and Path(artifact_path).exists() and not force_refit:
        print(f"Loading MS rolling fit from {artifact_path}")
        result = load_ms(artifact_path)
    else:
        result = fit_rolling_ms(
            returns,
            window=window,
            step=step,
            n_components=n_components,
            n_clusters=n_clusters,
            sticky_diag=sticky_diag,
            hmm_eps=hmm_eps,
            emission=emission,
        )
        if artifact_path:
            save_ms(result, artifact_path)
            print(f"Saved MS rolling fit to {artifact_path}")

    T_out = result.means_hist.shape[0]
    oos_returns = returns[window : window + T_out]

    portfolios_div    = generate_diversified(n_portfolios, D)
    portfolios_nondiv = generate_non_diversified(n_portfolios, D)

    # Student-t emission requires a different VaR solver (t CDF instead of Normal)
    var_fn = (
        compute_var_multi_level_student
        if emission == "student"
        else compute_var_multi_level
    )

    summaries: dict[str, list[pd.DataFrame]] = {
        "diversified": [],
        "non_diversified": [],
    }

    for alpha in alphas:
        vars_div = var_fn(result, portfolios_div, [alpha])[alpha]
        summaries["diversified"].append(
            aggregate_results(portfolios_div, oos_returns, vars_div, alpha)
        )
        vars_nondiv = var_fn(result, portfolios_nondiv, [alpha])[alpha]
        summaries["non_diversified"].append(
            aggregate_results(portfolios_nondiv, oos_returns, vars_nondiv, alpha)
        )

    return {
        "diversified": pd.concat(summaries["diversified"]),
        "non_diversified": pd.concat(summaries["non_diversified"]),
        "result": result,
    }


def run_ms_fixed_nu(
    returns: np.ndarray,
    nu: float | dict[float, float],
    window: int = 350,
    step: int = 1,
    n_components: int = 3,
    n_clusters: int = 2,
    n_portfolios: int = 200,
    alphas: list[float] | None = None,
    artifact_path: str | None = None,
    force_refit: bool = False,
    sticky_diag: float = 2.0,
    hmm_eps: float = 0.05,
) -> dict[str, pd.DataFrame | MSRollingResult]:
    """MS-mPPCA with Normal emissions but Student-t VaR at a fixed ν.

    Loads (or fits) the Normal-emission MS-mPPCA artifact, then computes VaR
    using t_ν quantiles instead of Gaussian quantiles.  No re-fitting needed
    when the Normal artifact is already cached.

    Parameters
    ----------
    nu : fixed degrees of freedom for the t VaR (calibrate with calibrate_nu())
    """
    if alphas is None:
        alphas = [0.05, 0.01]

    D = returns.shape[1]

    result: MSRollingResult
    if artifact_path and Path(artifact_path).exists() and not force_refit:
        print(f"Loading MS rolling fit from {artifact_path}")
        result = load_ms(artifact_path)
    else:
        result = fit_rolling_ms(
            returns,
            window=window,
            step=step,
            n_components=n_components,
            n_clusters=n_clusters,
            sticky_diag=sticky_diag,
            hmm_eps=hmm_eps,
            emission="normal",
        )
        if artifact_path:
            save_ms(result, artifact_path)
            print(f"Saved MS rolling fit to {artifact_path}")

    T_out       = result.means_hist.shape[0]
    oos_returns = returns[window : window + T_out]

    portfolios_div    = generate_diversified(n_portfolios, D)
    portfolios_nondiv = generate_non_diversified(n_portfolios, D)

    all_vars_div   = compute_var_multi_level_fixed_nu(result, portfolios_div,    alphas, nu)
    all_vars_nondiv = compute_var_multi_level_fixed_nu(result, portfolios_nondiv, alphas, nu)

    summaries: dict[str, list[pd.DataFrame]] = {
        "diversified": [],
        "non_diversified": [],
    }
    for alpha in alphas:
        summaries["diversified"].append(
            aggregate_results(portfolios_div, oos_returns, all_vars_div[alpha], alpha)
        )
        summaries["non_diversified"].append(
            aggregate_results(portfolios_nondiv, oos_returns, all_vars_nondiv[alpha], alpha)
        )

    return {
        "diversified": pd.concat(summaries["diversified"]),
        "non_diversified": pd.concat(summaries["non_diversified"]),
        "result": result,
    }
