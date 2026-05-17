"""Akaike Information Criterion for mPPCA model selection."""

import numpy as np


def aic(llh: float, d: int, q: int, k: int) -> float:
    """AIC for a single window.

    Parameters
    ----------
    llh : log-likelihood at convergence
    d   : data dimension D
    q   : latent dimension
    k   : number of mixture components
    """
    n_params = k * (d * q + 1 + d) + (k - 1)
    return 2 * n_params - 2 * llh


def aic_history(llh_hist: np.ndarray, d: int, q: int, k: int) -> np.ndarray:
    """AIC for every window in a rolling fit."""
    return np.array([aic(l, d, q, k) for l in llh_hist])
