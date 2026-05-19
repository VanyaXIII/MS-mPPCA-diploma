"""Download and cache S&P 500 stock price data via yfinance."""

import os
from pathlib import Path

import pandas as pd
import yfinance as yf

_DATA_DIR = Path(__file__).parents[2] / "data"
_CACHE_FILE = _DATA_DIR / "sp500_adj_close.csv"

# Manually curated subset of S&P 500 tickers across all 11 GICS sectors.
# Chosen for data availability over the full 2005-2024 window.
SP500_TICKERS = [
    # Communication Services
    "GOOGL", "META", "VZ", "T", "DIS", "NFLX", "CMCSA", "TMUS",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
    # Consumer Staples
    "WMT", "PG", "KO", "PEP", "COST", "CL", "MDLZ", "KHC",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C",
    # Health Care
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
    # Industrials
    "CAT", "BA", "HON", "UPS", "RTX", "GE", "LMT", "DE",
    # Information Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "QCOM", "TXN",
    # Materials
    "LIN", "APD", "ECL", "NEM", "FCX", "ALB", "DD", "PPG",
    # Real Estate
    "PLD", "AMT", "EQIX", "CCI", "SPG", "O", "DLR", "PSA",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
]


def download(
    tickers: list[str] | None = None,
    start: str = "2005-01-01",
    end: str = "2024-12-31",
    cache: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted closing prices, (date × ticker).

    Parameters
    ----------
    tickers:
        List of ticker symbols. Defaults to the built-in S&P 500 subset.
    start, end:
        Date range in YYYY-MM-DD format.
    cache:
        When True, save/load from ``data/sp500_adj_close.csv``.
    """
    _DATA_DIR.mkdir(exist_ok=True)
    if tickers is None:
        tickers = SP500_TICKERS

    if cache and _CACHE_FILE.exists():
        print(f"Loading cached data from {_CACHE_FILE}")
        return pd.read_csv(_CACHE_FILE, index_col=0, parse_dates=True)

    print(f"Downloading {len(tickers)} tickers from {start} to {end}…")
    raw: pd.DataFrame | None = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
    )
    assert raw is not None and not raw.empty, "yf.download returned no data"
    prices = pd.DataFrame(raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw)
    prices = prices.dropna(axis=1, thresh=int(0.9 * len(prices)))

    if cache:
        prices.to_csv(_CACHE_FILE)
        print(f"Saved to {_CACHE_FILE}")

    return prices
