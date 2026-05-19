"""Numba JIT utility functions shared across modules."""

import numpy as np
from math import exp, sqrt, pi, erf

from numba import jit


@jit(nopython=True)
def mean_row(arr: np.ndarray) -> np.ndarray:
    nrows, ncols = arr.shape
    mean_vals = np.zeros(ncols)
    for i in range(ncols):
        col_sum = 0.0
        for j in range(nrows):
            col_sum += arr[j, i]
        mean_vals[i] = col_sum / nrows
    return mean_vals


@jit(nopython=True)
def std_row(arr: np.ndarray) -> np.ndarray:
    nrows, ncols = arr.shape
    mean_vals = np.zeros(ncols)
    std_vals = np.zeros(ncols)
    for i in range(ncols):
        col_sum = 0.0
        for j in range(nrows):
            col_sum += arr[j, i]
        mean_vals[i] = col_sum / nrows
    for i in range(ncols):
        var_sum = 0.0
        for j in range(nrows):
            var_sum += (arr[j, i] - mean_vals[i]) ** 2
        std_vals[i] = sqrt(var_sum / nrows)
    return std_vals


@jit(nopython=True)
def logsumexp_rows(arr: np.ndarray) -> np.ndarray:
    """Log-sum-exp along axis=1 for numerical stability."""
    n = arr.shape[0]
    result = np.empty(n)
    for i in range(n):
        max_val = arr[i, 0]
        for j in range(1, arr.shape[1]):
            if arr[i, j] > max_val:
                max_val = arr[i, j]
        s = 0.0
        for j in range(arr.shape[1]):
            s += exp(arr[i, j] - max_val)
        result[i] = np.log(s) + max_val
    return result


@jit(nopython=True)
def max_per_row(arr: np.ndarray) -> np.ndarray:
    nrows = arr.shape[0]
    max_vals = np.empty(nrows)
    for i in range(nrows):
        max_val = arr[i, 0]
        for j in range(1, arr.shape[1]):
            if arr[i, j] > max_val:
                max_val = arr[i, j]
        max_vals[i] = max_val
    return max_vals


@jit(nopython=True)
def norm_cdf(x: float, mean: float, std: float) -> float:
    z = (x - mean) / std
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


@jit(nopython=True)
def norm_pdf(x: float, mean: float, std: float) -> float:
    u = (x - mean) / abs(std)
    return exp(-u * u / 2.0) / (sqrt(2.0 * pi) * abs(std))


@jit(nopython=True)
def mixture_cdf(x: float, means: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> float:
    result = 0.0
    for i in range(len(means)):
        result += weights[i] * norm_cdf(x, means[i], stds[i])
    return result


@jit(nopython=True)
def mixture_pdf(x: float, means: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> float:
    result = 0.0
    for i in range(len(means)):
        result += weights[i] * norm_pdf(x, means[i], stds[i])
    return result


@jit(nopython=True)
def scalar_norm_ppf(p: float, mean: float, std: float) -> float:
    if p < 0.0 or p > 1.0:
        return np.nan
    lo, hi = mean - 10.0 * std, mean + 10.0 * std
    while hi - lo > 1e-5:
        mid = (lo + hi) / 2.0
        if norm_cdf(mid, mean, std) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@jit(nopython=True)
def find_quantile_bounds(
    alpha: float, means: np.ndarray, stds: np.ndarray
) -> tuple[float, float]:
    quantiles = np.empty(len(means))
    for i in range(len(means)):
        quantiles[i] = scalar_norm_ppf(alpha, means[i], stds[i])
    return np.min(quantiles), np.max(quantiles)


@jit(nopython=True)
def bisection_mixture_quantile(
    alpha: float,
    lo: float,
    hi: float,
    means: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
    tol: float = 1e-4,
) -> float:
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if mixture_cdf(mid, means, stds, weights) < alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@jit(nopython=True)
def newton_raphson_mixture_quantile(
    alpha: float,
    x0: float,
    means: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 2000,
) -> float:
    x = x0
    for _ in range(max_iter):
        fx = mixture_cdf(x, means, stds, weights) - alpha
        fpx = mixture_pdf(x, means, stds, weights)
        x_new = x - fx / fpx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x
