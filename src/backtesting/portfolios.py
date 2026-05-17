"""Generate diversified and non-diversified portfolio weight vectors."""

import numpy as np


def generate_non_diversified(
    n_portfolios: int = 200,
    n_stocks: int = 35,
    random_state: int = 57,
) -> np.ndarray:
    """Return (n_portfolios, n_stocks) weight matrix.

    Each portfolio concentrates weight on 2–5 randomly chosen stocks.
    """
    rng = np.random.default_rng(random_state)
    portfolios = np.zeros((n_portfolios, n_stocks))
    for i in range(n_portfolios):
        n_dom = rng.integers(2, 6)
        idx = rng.choice(n_stocks, size=n_dom, replace=False)
        portfolios[i, idx] = rng.dirichlet(np.ones(n_dom))
        portfolios[i] /= portfolios[i].sum()
    return portfolios


def generate_diversified(
    n_portfolios: int = 200,
    n_stocks: int = 35,
    noise_scale: float | None = None,
    random_state: int = 57,
) -> np.ndarray:
    """Return (n_portfolios, n_stocks) weight matrix.

    Each portfolio is near-equal-weight with small Gaussian noise.
    """
    rng = np.random.default_rng(random_state)
    if noise_scale is None:
        noise_scale = 1.0 / (n_stocks * 5)
    base = np.ones(n_stocks) / n_stocks
    portfolios = np.zeros((n_portfolios, n_stocks))
    for i in range(n_portfolios):
        w = base + rng.normal(0, noise_scale, n_stocks)
        w = np.clip(w, 0, None)
        portfolios[i] = w / w.sum()
    return portfolios
