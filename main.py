import argparse
from pathlib import Path

import pandas as pd

from src.backtesting.pipeline import run, run_ms, run_ms_fixed_nu
from src.models.var import calibrate_nu_per_alpha
from src.data.loader import download
from src.data.preprocessing import get_returns_array

DATA_DIR         = Path("data")
RUN_DIR          = DATA_DIR / "run"
VANILLA_ARTIFACT = str(RUN_DIR / "rolling_fit_vanilla.npz")
MS_ARTIFACT      = str(RUN_DIR / "rolling_fit_ms.npz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mPPCA rolling VaR pipeline")
    p.add_argument("--window",     type=int,   default=350,          help="Rolling window length (default 350)")
    p.add_argument("--step",       type=int,   default=1,            help="Window stride (default 1)")
    p.add_argument("--components", type=int,   default=3,            help="Latent dimension q (default 3)")
    p.add_argument("--clusters",   type=int,   default=2,            help="Mixture components K (default 2)")
    p.add_argument("--portfolios", type=int,   default=200,          help="Portfolios per type (default 200)")
    p.add_argument("--alphas",     type=float, nargs="+", default=[0.05, 0.01], help="VaR levels")
    p.add_argument("--start",      default="2005-01-01",             help="Data start date")
    p.add_argument("--end",        default="2024-12-31",             help="Data end date")
    p.add_argument("--no-download",    action="store_true", help="Use cached data")
    p.add_argument("--force-download", action="store_true", help="Force fresh download")
    p.add_argument("--force-refit",    action="store_true", help="Re-fit even if artifact exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    prices = download(start=args.start, end=args.end, cache=not args.force_download)
    print(f"Prices: {prices.shape[0]} days × {prices.shape[1]} tickers")

    returns, _, _ = get_returns_array(prices)
    print(f"Returns: {returns.shape[0]} days × {returns.shape[1]} assets after cleaning")

    shared = dict(
        window=args.window,
        step=args.step,
        n_components=args.components,
        n_clusters=args.clusters,
        n_portfolios=args.portfolios,
        alphas=args.alphas,
        force_refit=args.force_refit,
    )

    print("\n--- Vanilla mPPCA (Normal) ---")
    vanilla_dict = run(returns=returns, artifact_path=VANILLA_ARTIFACT, **shared)

    print("\n--- MS-mPPCA (Normal) ---")
    ms_dict = run_ms(returns=returns, artifact_path=MS_ARTIFACT, emission="normal", **shared)

    ms_result = ms_dict["result"]
    T_out     = ms_result.means_hist.shape[0]
    oos_cal   = returns[args.window : args.window + T_out]
    print("\nCalibrating per-alpha ν...")
    nu_dict = calibrate_nu_per_alpha(ms_result, oos_cal, args.alphas)

    nu_label = " | ".join(f"α={a:.0%} ν={nu_dict[a]:.1f}" for a in args.alphas)
    print(f"\n--- MS-mPPCA + per-α Student-t VaR  ({nu_label}) ---")
    ms_tvar_dict = run_ms_fixed_nu(returns=returns, nu=nu_dict, artifact_path=MS_ARTIFACT, **shared)

    COLS = ["breach_rate", "pvalue", "pvalue_pass", "ci_95_pass", "ci_99_pass", "ind_pass"]
    COL_RENAME = {
        "breach_rate": "Breach rate",
        "pvalue"     : "Kupiec p-val",
        "pvalue_pass": "Kupiec pass",
        "ci_95_pass" : "CI 95%",
        "ci_99_pass" : "CI 99%",
        "ind_pass"   : "Christ. pass",
    }
    MODELS = [
        ("Vanilla    ", vanilla_dict),
        ("MS-Normal  ", ms_dict),
        ("MS-t(per-α)", ms_tvar_dict),
    ]

    for port_type, label in (("diversified", "Diversified"), ("non_diversified", "Concentrated")):
        print(f"\n{'=' * 78}")
        print(f"  Comparative backtest — {label} portfolios")
        print("=" * 78)
        rows = []
        for alpha in args.alphas:
            for name, d in MODELS:
                df  = d[port_type][COLS].rename(columns=COL_RENAME)
                row = df[df.index == f"alpha={alpha}"].copy()
                assert len(row) == 1
                row.index = [f"{name}  α={alpha}"]
                rows.append(row)
        print(pd.concat(rows).round(4).to_string())

    print(f"\nDone.  Artifacts in {RUN_DIR}/")


if __name__ == "__main__":
    main()
