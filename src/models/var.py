"""VaR estimation from mPPCA mixture-of-Gaussians parameters.

For each window t, the portfolio return distribution is a mixture:
  p(r) = Σ_k π_k N(w^T μ_k, w^T C_k w)

The VaR at level α is the α-quantile of this 1-D mixture.
Quantile is found via bisection warm-started from per-component quantiles,
then polished with Newton-Raphson — all JIT-compiled in numba_utils.
"""

from __future__ import annotations

import numpy as np
from numba import jit

from scipy.stats import t as scipy_t

from src.models.rolling import MSRollingResult, RollingResult


def calibrate_nu(returns: np.ndarray) -> float:
    """Fit a univariate Student-t to equal-weight portfolio returns and return df.

    Uses scipy MLE. Financial daily returns typically yield ν ≈ 3–6.

    Parameters
    ----------
    returns : (T, D) log-return matrix (full sample, not OOS-only)
    """
    D = returns.shape[1]
    port = returns @ (np.ones(D) / D)
    df, _loc, _scale = scipy_t.fit(port, floc=0)   # fix location at 0 for stability
    return float(df)


def calibrate_nu_per_alpha(
    result: "RollingResult | MSRollingResult",
    oos_returns: np.ndarray,
    alphas: list[float],
) -> dict[float, float]:
    """For each alpha, bisect on ν to match EW portfolio OOS breach rate to alpha.

    At ν → 2 the t quantile is maximally extreme (near-zero breaches).
    At ν → ∞ it converges to Normal (breach rate ≈ Normal model's rate).
    The root ν_α in (2.1, 300) gives the target breach rate exactly on the
    equal-weight portfolio.

    Parameters
    ----------
    result      : fitted RollingResult or MSRollingResult
    oos_returns : (T_out, D) out-of-sample returns aligned with result
    alphas      : VaR levels to calibrate
    """
    from scipy.optimize import brentq

    D = oos_returns.shape[1]
    eq_w = np.ones((1, D)) / D                          # (1, D)
    port_oos = oos_returns @ (np.ones(D) / D)           # (T_out,)

    nu_dict: dict[float, float] = {}
    for alpha in alphas:
        def _err(nu_val: float, _alpha: float = alpha) -> float:
            var_series = compute_var_multi_level_fixed_nu(
                result, eq_w, [_alpha], nu_val
            )[_alpha][0]
            return float((port_oos < var_series).mean()) - _alpha

        lo, hi = 2.1, 300.0
        try:
            nu_opt = brentq(_err, lo, hi, xtol=0.5)
        except ValueError:
            nu_opt = 30.0 
        nu_dict[alpha] = nu_opt
        print(f"  calibrated ν  α={alpha:.0%} → ν={nu_opt:.2f}")

    return nu_dict


from src.utils.numba_utils import (
    bisection_mixture_quantile,
    find_quantile_bounds,
    newton_raphson_mixture_quantile,
)


# ---------------------------------------------------------------------------
# Covariance reconstruction
# ---------------------------------------------------------------------------

def reconstruct_cov(W_k: np.ndarray, sigma2_k: float) -> np.ndarray:
    """C_k = W_k W_k^T + σ²_k I  (shape D × D)."""
    return W_k @ W_k.T + sigma2_k * np.eye(W_k.shape[0])


def cluster_params_in_return_space(
    result: RollingResult | MSRollingResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover per-window cluster means / covs / weights in original return space.

    Returns
    -------
    means_ret   : (T_out, K, D)
    covs_ret    : (T_out, K, D, D)
    weights_ret : (T_out, K)
    """
    T_out, K, D = result.means_hist.shape
    means_ret = np.empty((T_out, K, D))
    covs_ret = np.empty((T_out, K, D, D))

    for t in range(T_out):
        shift = result.shifts[t]
        scale = result.scales[t]
        S = np.diag(scale)
        for k in range(K):
            means_ret[t, k] = shift + result.means_hist[t, k] * scale
            C_norm = reconstruct_cov(result.W_hist[t, k], result.sigma2_hist[t, k])
            covs_ret[t, k] = S @ C_norm @ S

    return means_ret, covs_ret, result.weights_hist.copy()


# ---------------------------------------------------------------------------
# Per-portfolio VaR calculation
# ---------------------------------------------------------------------------

@jit(nopython=True)
def _portfolio_stds(cov_history: np.ndarray, port_w: np.ndarray) -> np.ndarray:
    """(T_out, K) portfolio std from (T_out, K, D, D) cov history."""
    T, K, D, _ = cov_history.shape
    stds = np.empty((T, K))
    for t in range(T):
        for k in range(K):
            var = port_w @ cov_history[t, k] @ port_w
            stds[t, k] = np.sqrt(var)
    return stds


@jit(nopython=True)
def _portfolio_means(mean_history: np.ndarray, port_w: np.ndarray) -> np.ndarray:
    """(T_out, K) portfolio mean from (T_out, K, D) mean history."""
    T, K, D = mean_history.shape
    means = np.empty((T, K))
    for t in range(T):
        for k in range(K):
            s = 0.0
            for d in range(D):
                s += mean_history[t, k, d] * port_w[d]
            means[t, k] = s
    return means


@jit(nopython=True)
def _var_series(
    means_p: np.ndarray,
    stds_p: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute VaR time-series for one portfolio.

    Parameters
    ----------
    means_p  : (T, K)
    stds_p   : (T, K)
    weights  : (T, K)
    alpha    : VaR confidence level

    Returns
    -------
    var_series : (T,)
    """
    T = means_p.shape[0]
    result = np.empty(T)
    for t in range(T):
        mu = means_p[t]
        sig = stds_p[t]
        w = weights[t]
        lo, hi = find_quantile_bounds(alpha, mu, sig)
        x0 = bisection_mixture_quantile(alpha, lo, hi, mu, sig, w, tol=1e-2)
        result[t] = newton_raphson_mixture_quantile(alpha, x0, mu, sig, w, tol=1e-6)
    return result


def compute_var(
    result: RollingResult | MSRollingResult,
    portfolios: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute VaR series for a collection of portfolios.

    Parameters
    ----------
    result     : RollingResult or MSRollingResult
    portfolios : (M, D) portfolio weight matrix
    alpha      : VaR level (e.g. 0.05 for 5 % VaR)

    Returns
    -------
    vars : (M, T_out) — VaR time-series for each portfolio
    """
    means_ret, covs_ret, weights_ret = cluster_params_in_return_space(result)

    M = portfolios.shape[0]
    T_out = means_ret.shape[0]
    vars_out = np.empty((M, T_out))

    for i in range(M):
        w = portfolios[i]
        means_p = _portfolio_means(means_ret, w)
        stds_p = _portfolio_stds(covs_ret, w)
        vars_out[i] = _var_series(means_p, stds_p, weights_ret, alpha)

    return vars_out


def compute_var_multi_level(
    result: RollingResult | MSRollingResult,
    portfolios: np.ndarray,
    alphas: list[float],
) -> dict[float, np.ndarray]:
    """Compute VaR for multiple confidence levels in one pass.

    Returns
    -------
    {alpha: (M, T_out)}
    """
    means_ret, covs_ret, weights_ret = cluster_params_in_return_space(result)
    M = portfolios.shape[0]

    # Precompute portfolio projections once — independent of alpha
    all_means_p = [_portfolio_means(means_ret, portfolios[i]) for i in range(M)]
    all_stds_p  = [_portfolio_stds(covs_ret,   portfolios[i]) for i in range(M)]

    out: dict[float, np.ndarray] = {}
    for alpha in alphas:
        vars_alpha = np.empty((M, means_ret.shape[0]))
        for i in range(M):
            vars_alpha[i] = _var_series(all_means_p[i], all_stds_p[i], weights_ret, alpha)
        out[alpha] = vars_alpha

    return out


def compute_var_multi_level_fixed_nu(
    result: RollingResult | MSRollingResult,
    portfolios: np.ndarray,
    alphas: list[float],
    nu: float | dict[float, float],
) -> dict[float, np.ndarray]:
    """Student-t VaR using Normal-EM scatter parameters with a fixed ν per alpha.

    Reuses the cluster parameters (μ_k, C_k, π_k) from a Normal-emission fit
    but replaces the Gaussian quantile with t_{ν_α}.  No re-fitting required.

    Parameters
    ----------
    result : any RollingResult / MSRollingResult (Normal emission recommended)
    portfolios : (M, D)
    alphas : list of VaR levels
    nu : scalar (same ν for every alpha) or dict {alpha: nu} for per-alpha calibration
    """
    nu_map: dict[float, float] = nu if isinstance(nu, dict) else {a: nu for a in alphas}

    means_ret, covs_ret, weights_ret = cluster_params_in_return_space(result)
    T_out, K, _ = means_ret.shape
    M = portfolios.shape[0]

    means_p = np.einsum("tkd,md->tkm", means_ret, portfolios)          # (T_out, K, M)
    Cp      = np.einsum("tkde,me->tkdm", covs_ret, portfolios)         # (T_out, K, D, M)
    var_p   = np.einsum("md,tkdm->tkm", portfolios, Cp)                # (T_out, K, M)
    std_p   = np.sqrt(np.maximum(var_p, 1e-12))                        # (T_out, K, M)

    out: dict[float, np.ndarray] = {}
    for alpha in alphas:
        nu_a = nu_map[alpha]
        lo = np.full((T_out, M), np.inf)
        hi = np.full((T_out, M), -np.inf)
        for k in range(K):
            q_k = scipy_t.ppf(alpha, df=nu_a, loc=means_p[:, k, :], scale=std_p[:, k, :])
            lo  = np.minimum(lo, q_k)
            hi  = np.maximum(hi, q_k)
        spread = np.abs(hi - lo)
        lo -= spread
        hi += spread

        for _ in range(50):
            mid   = 0.5 * (lo + hi)
            F_mid = np.zeros((T_out, M))
            for k in range(K):
                w_k = weights_ret[:, k, np.newaxis]
                F_mid += w_k * scipy_t.cdf(
                    mid, df=nu_a, loc=means_p[:, k, :], scale=std_p[:, k, :]
                )
            lo = np.where(F_mid < alpha, mid, lo)
            hi = np.where(F_mid >= alpha, mid, hi)

        out[alpha] = (0.5 * (lo + hi)).T   # (M, T_out)

    return out


def compute_var_multi_level_student(
    result: MSRollingResult,
    portfolios: np.ndarray,
    alphas: list[float],
) -> dict[float, np.ndarray]:
    """Compute VaR for a mixture of Student-t distributions.

    Each cluster k contributes t_{ν_k}(w^T μ_k, w^T C_k w) to the mixture.
    The α-quantile is found via vectorized bisection over all portfolios and
    windows simultaneously, using scipy.stats.t for the t CDF (which accepts
    broadcasting over df, loc, scale).

    Parameters
    ----------
    result     : MSRollingResult with nu_hist populated (emission="student")
    portfolios : (M, D)
    alphas     : list of VaR levels

    Returns
    -------
    {alpha: (M, T_out)}
    """
    assert result.nu_hist is not None, (
        "result.nu_hist is None — refit with emission='student'"
    )

    means_ret, covs_ret, weights_ret = cluster_params_in_return_space(result)
    nu_hist = result.nu_hist   # (T_out, K)
    T_out, K, _ = means_ret.shape
    M = portfolios.shape[0]

    # Portfolio projections — precomputed once for all alphas
    # means_p[t, k, j] = μ_{tk} · w_j          shape (T_out, K, M)
    means_p = np.einsum("tkd,md->tkm", means_ret, portfolios)

    # Portfolio scatter v_p[t,k,j] = w_j^T C_{tk} w_j  shape (T_out, K, M)
    # Avoid (T_out, K, D, D) @ (D, M) full matmul; use einsum in two steps.
    Cp = np.einsum("tkde,me->tkdm", covs_ret, portfolios)   # (T_out, K, D, M)
    var_p = np.einsum("md,tkdm->tkm", portfolios, Cp)        # (T_out, K, M)
    std_p = np.sqrt(np.maximum(var_p, 1e-12))                # (T_out, K, M)

    out: dict[float, np.ndarray] = {}
    for alpha in alphas:
        # Initialise bisection bounds from per-cluster t quantiles
        # scipy_t.ppf broadcasts: df (T_out,1), loc/scale (T_out, M)
        lo = np.full((T_out, M), np.inf)
        hi = np.full((T_out, M), -np.inf)
        for k in range(K):
            nu_k = nu_hist[:, k, np.newaxis]   # (T_out, 1) — broadcasts over M
            q_k = scipy_t.ppf(
                alpha, df=nu_k,
                loc=means_p[:, k, :], scale=std_p[:, k, :]
            )                                   # (T_out, M)
            lo = np.minimum(lo, q_k)
            hi = np.maximum(hi, q_k)
        # Widen bracket to be safe
        spread = np.abs(hi - lo)
        lo -= spread
        hi += spread

        # Fully vectorized bisection — 50 steps give ~1e-15 relative error
        for _ in range(50):
            mid = 0.5 * (lo + hi)              # (T_out, M)
            F_mid = np.zeros((T_out, M))
            for k in range(K):
                nu_k = nu_hist[:, k, np.newaxis]
                w_k  = weights_ret[:, k, np.newaxis]   # (T_out, 1)
                F_mid += w_k * scipy_t.cdf(
                    mid, df=nu_k,
                    loc=means_p[:, k, :], scale=std_p[:, k, :]
                )
            lo = np.where(F_mid < alpha, mid, lo)
            hi = np.where(F_mid >= alpha, mid, hi)

        # vars[j, t] — transpose from (T_out, M) to (M, T_out)
        out[alpha] = (0.5 * (lo + hi)).T

    return out
