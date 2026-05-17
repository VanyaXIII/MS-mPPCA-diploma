"""Data preprocessing: cleaning, log-returns, normalization helpers."""

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns from adjusted closing prices.

    Returns a DataFrame with one fewer row than ``prices``.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna() # type: ignore
    return log_returns


def clean(
    prices: pd.DataFrame,
    min_history_frac: float = 0.9,
) -> pd.DataFrame:
    """Drop tickers with too many NaNs and forward-fill residual gaps.

    Parameters
    ----------
    prices:
        Adjusted closing prices (date × ticker).
    min_history_frac:
        Minimum fraction of non-NaN rows required to keep a ticker.
    """
    threshold = int(min_history_frac * len(prices))
    prices = prices.dropna(axis=1, thresh=threshold)
    prices = prices.ffill().dropna()
    return prices


def get_returns_array(
    prices: pd.DataFrame,
    min_history_frac: float = 0.9,
) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """Full preprocessing pipeline: clean → log-returns → numpy array.

    Returns
    -------
    values : ndarray, shape (T, D)
        Log-return matrix.
    dates : DatetimeIndex
        Corresponding trading dates.
    tickers : list[str]
        Column names after cleaning.
    """
    prices = clean(prices, min_history_frac)
    log_returns = compute_log_returns(prices)
    return log_returns.values, log_returns.index, list(log_returns.columns) #type: ignore


def normalize_window(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-mean, unit-std normalize a (T, D) window.

    Returns
    -------
    normalized : ndarray
    shift : ndarray, shape (D,)
    scale : ndarray, shape (D,)
    """
    shift = data.mean(axis=0)
    scale = data.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (data - shift) / scale, shift, scale


def denormalize_mean(
    mean_normalized: np.ndarray, shift: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Map cluster mean from normalized space back to return space."""
    return shift + mean_normalized * scale


def denormalize_cov(
    cov_normalized: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Map cluster covariance from normalized space back to return space."""
    S = np.diag(scale)
    return S @ cov_normalized @ S
