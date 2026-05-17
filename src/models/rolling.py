"""Sliding-window mPPCA and MS-mPPCA fitting with warm-start.

Two independent pipelines — use fit_rolling for classic mPPCA,
fit_rolling_ms for the Markov Switching coupled version.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from src.data.preprocessing import normalize_window
from src.models.mppca import fit, init_kmeans_pca
from src.models.ms_mppca import align_labels, fit_ms, init_ms_from_mppca
from src.models.hmm import stationary
from src.utils.aic import aic


@dataclass
class RollingResult:
    """Packed history of rolling mPPCA fits.

    Shapes
    ------
    means_hist   : (T_out, K, D)
    W_hist       : (T_out, K, D, q)
    sigma2_hist  : (T_out, K)
    weights_hist : (T_out, K)
    R_hist       : (T_out, T_window, K)  — responsibilities per window (for MS coupling)
    llh_hist     : (T_out,)
    shifts       : (T_out, D)
    scales       : (T_out, D)
    """
    means_hist: np.ndarray
    W_hist: np.ndarray
    sigma2_hist: np.ndarray
    weights_hist: np.ndarray
    R_hist: list[np.ndarray]
    llh_hist: np.ndarray
    shifts: np.ndarray
    scales: np.ndarray


def fit_rolling(
    returns: np.ndarray,
    window: int = 350,
    step: int = 1,
    n_components: int = 3,
    n_clusters: int = 2,
    n_iter_init: int = 5000,
    tol_init: float = 1e-5,
    n_iter: int = 500,
    tol: float = 1e-2,
    random_state: int = 42,
) -> RollingResult:
    """Fit mPPCA on every rolling window of ``returns``.

    Parameters
    ----------
    returns      : (T, D) log-return matrix
    window       : length of each fitting window (δ)
    step         : stride between windows (usually 1)
    n_components : latent dimension q
    n_clusters   : number of mixture components K
    n_iter_init  : EM iterations for the first window (cold start)
    tol_init     : convergence tol for the first window
    n_iter       : EM iterations for subsequent windows (warm start)
    tol          : convergence tol for subsequent windows
    """
    T, D = returns.shape
    n_windows = (T - window) // step

    means_list: list[np.ndarray] = []
    W_list: list[np.ndarray] = []
    sigma2_list: list[np.ndarray] = []
    weights_list: list[np.ndarray] = []
    R_list: list[np.ndarray] = []
    llh_list: list[float] = []
    shifts_list: list[np.ndarray] = []
    scales_list: list[np.ndarray] = []

    means = W = sigma2 = weights = None

    for i in tqdm(range(n_windows), desc="Rolling mPPCA"):
        t0 = i * step
        data_raw = returns[t0 : t0 + window]
        data, shift, scale = normalize_window(data_raw)

        if means is None:
            means, W, sigma2, weights = init_kmeans_pca(
                data, n_components, n_clusters, random_state
            )
            n_it, conv_tol = n_iter_init, tol_init
        else:
            n_it, conv_tol = n_iter, tol
        assert W is not None and sigma2 is not None and weights is not None, "params have not been trained"
        means, W, sigma2, weights, R, llh_history = fit(
            data, means, W, sigma2, weights, n_components,
            n_iter=n_it, tol=conv_tol,
        )

        means_list.append(means.copy())
        W_list.append(W.copy())
        sigma2_list.append(sigma2.copy())
        weights_list.append(weights.copy())
        R_list.append(R.copy())
        llh_list.append(llh_history[-1])
        shifts_list.append(shift.copy())
        scales_list.append(scale.copy())

    return RollingResult(
        means_hist=np.array(means_list),
        W_hist=np.array(W_list),
        sigma2_hist=np.array(sigma2_list),
        weights_hist=np.array(weights_list),
        R_hist=R_list,
        llh_hist=np.array(llh_list),
        shifts=np.array(shifts_list),
        scales=np.array(scales_list),
    )


def save(result: RollingResult, path: str) -> None:
    np.savez_compressed(
        path,
        means_hist=result.means_hist,
        W_hist=result.W_hist,
        sigma2_hist=result.sigma2_hist,
        weights_hist=result.weights_hist,
        llh_hist=result.llh_hist,
        shifts=result.shifts,
        scales=result.scales,
    )


def load(path: str) -> RollingResult:
    data = np.load(path)
    return RollingResult(
        means_hist=data["means_hist"],
        W_hist=data["W_hist"],
        sigma2_hist=data["sigma2_hist"],
        weights_hist=data["weights_hist"],
        R_hist=[],  # not stored (large); recompute if needed
        llh_hist=data["llh_hist"],
        shifts=data["shifts"],
        scales=data["scales"],
    )


# ---------------------------------------------------------------------------
# MS-mPPCA rolling fit
# ---------------------------------------------------------------------------

@dataclass
class MSRollingResult:
    """Packed history of rolling MS-mPPCA fits.

    Shapes
    ------
    means_hist              : (T_out, K, D)
    W_hist                  : (T_out, K, D, q)
    sigma2_hist             : (T_out, K)
    A_hist                  : (T_out, K, K)   — transition matrices
    pi0_hist                : (T_out, K)       — initial distributions
    predicted_pi_next_hist  : (T_out, K)       — A^T gamma_T, used as VaR weights
    stationary_hist         : (T_out, K)       — long-run stationary of A
    weights_hist            : (T_out, K)       — alias for predicted_pi_next_hist
                                                 (keeps VaR code compatible)
    gamma_hist              : list of (T_window, K)  — not persisted to disk
    llh_hist                : (T_out,)
    shifts                  : (T_out, D)
    scales                  : (T_out, D)
    """
    means_hist: np.ndarray
    W_hist: np.ndarray
    sigma2_hist: np.ndarray
    A_hist: np.ndarray
    pi0_hist: np.ndarray
    predicted_pi_next_hist: np.ndarray
    stationary_hist: np.ndarray
    weights_hist: np.ndarray          # == predicted_pi_next_hist; keeps var.py unchanged
    gamma_hist: list[np.ndarray]
    llh_hist: np.ndarray
    shifts: np.ndarray
    scales: np.ndarray
    nu_hist: np.ndarray | None = None  # (T_out, K) degrees of freedom; None for normal emission


def fit_rolling_ms(
    returns: np.ndarray,
    window: int = 350,
    step: int = 1,
    n_components: int = 3,
    n_clusters: int = 2,
    n_iter_init: int = 5000,
    tol_init: float = 1e-5,
    n_iter_ms_init: int = 200,
    n_iter_ms: int = 100,
    tol_ms: float = 1e-3,
    hmm_eps: float = 1e-4,
    sticky_diag: float = 0.1,
    emission: str = "normal",
    random_state: int = 42,
) -> MSRollingResult:
    """Fit MS-mPPCA on every rolling window of ``returns``.

    Strategy per window
    -------------------
    Window 0 (cold start):
      1. Run vanilla mPPCA to convergence → (μ, W, σ², R).
      2. Initialise A from soft co-occurrences in R.
      3. Run MS-mPPCA EM for n_iter_ms_init steps.

    Window i > 0 (warm start):
      - Carry (μ, W, σ², A) from previous window.
      - Set π_0 = gamma[1] from previous window (one-step propagation).
      - Run MS-mPPCA EM for n_iter_ms steps.
      - Align cluster labels to previous window's means.

    Parameters
    ----------
    n_iter_ms_init : MS-mPPCA iterations for window 0
    n_iter_ms      : MS-mPPCA iterations for subsequent windows
    hmm_eps        : Dirichlet smoothing for transition matrix rows
    sticky_diag    : extra mass on A diagonal at initialisation
    emission       : "normal" (Gaussian) or "student" (Student-t with learnable ν per cluster)
    """
    T, D = returns.shape
    n_windows = (T - window) // step

    means_list: list[np.ndarray] = []
    W_list: list[np.ndarray] = []
    sigma2_list: list[np.ndarray] = []
    A_list: list[np.ndarray] = []
    pi0_list: list[np.ndarray] = []
    pred_pi_list: list[np.ndarray] = []
    stat_list: list[np.ndarray] = []
    gamma_list: list[np.ndarray] = []
    nu_list: list[np.ndarray] = []
    llh_list: list[float] = []
    shifts_list: list[np.ndarray] = []
    scales_list: list[np.ndarray] = []

    means = W = sigma2 = A = pi0 = None
    prev_gamma: np.ndarray | None = None
    prev_means: np.ndarray | None = None
    # Student-t: initialise ν=5 for all clusters (warm-started across windows)
    nu: np.ndarray | None = np.full(n_clusters, 5.0) if emission == "student" else None

    for i in tqdm(range(n_windows), desc=f"Rolling MS-mPPCA ({emission})"):
        t0 = i * step
        data_raw = returns[t0 : t0 + window]
        data, shift, scale = normalize_window(data_raw)

        if means is None:
            # Cold start: vanilla mPPCA then init HMM
            means, W, sigma2, weights = init_kmeans_pca(
                data, n_components, n_clusters, random_state
            )
            means, W, sigma2, weights, R, _ = fit(
                data, means, W, sigma2, weights, n_components,
                n_iter=n_iter_init, tol=tol_init,
            )
            A, pi0 = init_ms_from_mppca(R, sticky_diag=sticky_diag)
            n_it_ms = n_iter_ms_init
        else:
            assert A is not None and pi0 is not None
            assert prev_gamma is not None
            pi0 = prev_gamma[1] if step == 1 else prev_gamma[min(step, len(prev_gamma) - 1)]
            pi0 = pi0 / pi0.sum()
            n_it_ms = n_iter_ms

        assert W is not None and sigma2 is not None
        means, W, sigma2, A, pi0, gamma, llh_history, nu = fit_ms(
            data, means, W, sigma2, A, pi0, n_components,
            n_iter=n_it_ms, tol=tol_ms, hmm_eps=hmm_eps,
            emission=emission, nu=nu,
        )

        # Align labels to previous window to prevent swapping regime indices
        if prev_means is not None:
            means, W, sigma2, A, pi0, gamma = align_labels(
                means, W, sigma2, A, pi0, gamma, prev_means
            )

        # One-step-ahead mixing weight: A^T γ_T
        pred_pi = A.T @ gamma[-1]
        pred_pi /= pred_pi.sum()
        stat = stationary(A)

        means_list.append(means.copy())
        W_list.append(W.copy())
        sigma2_list.append(sigma2.copy())
        A_list.append(A.copy())
        pi0_list.append(pi0.copy())
        pred_pi_list.append(pred_pi.copy())
        stat_list.append(stat.copy())
        gamma_list.append(gamma.copy())
        if nu is not None:
            nu_list.append(nu.copy())
        llh_list.append(llh_history[-1])
        shifts_list.append(shift.copy())
        scales_list.append(scale.copy())

        prev_gamma = gamma
        prev_means = means.copy()

    predicted_pi = np.array(pred_pi_list)
    return MSRollingResult(
        means_hist=np.array(means_list),
        W_hist=np.array(W_list),
        sigma2_hist=np.array(sigma2_list),
        A_hist=np.array(A_list),
        pi0_hist=np.array(pi0_list),
        predicted_pi_next_hist=predicted_pi,
        stationary_hist=np.array(stat_list),
        weights_hist=predicted_pi,
        gamma_hist=gamma_list,
        llh_hist=np.array(llh_list),
        shifts=np.array(shifts_list),
        scales=np.array(scales_list),
        nu_hist=np.array(nu_list) if nu_list else None,
    )


def save_ms(result: MSRollingResult, path: str) -> None:
    fixed = dict(
        means_hist=result.means_hist,
        W_hist=result.W_hist,
        sigma2_hist=result.sigma2_hist,
        A_hist=result.A_hist,
        pi0_hist=result.pi0_hist,
        predicted_pi_next_hist=result.predicted_pi_next_hist,
        stationary_hist=result.stationary_hist,
        llh_hist=result.llh_hist,
        shifts=result.shifts,
        scales=result.scales,
    )
    if result.nu_hist is not None:
        np.savez_compressed(path, nu_hist=result.nu_hist, **fixed)
    else:
        np.savez_compressed(path, **fixed)


def load_ms(path: str) -> MSRollingResult:
    d = np.load(path)
    predicted_pi = d["predicted_pi_next_hist"]
    return MSRollingResult(
        means_hist=d["means_hist"],
        W_hist=d["W_hist"],
        sigma2_hist=d["sigma2_hist"],
        A_hist=d["A_hist"],
        pi0_hist=d["pi0_hist"],
        predicted_pi_next_hist=predicted_pi,
        stationary_hist=d["stationary_hist"],
        weights_hist=predicted_pi,
        gamma_hist=[],
        llh_hist=d["llh_hist"],
        shifts=d["shifts"],
        scales=d["scales"],
        nu_hist=d["nu_hist"] if "nu_hist" in d else None,
    )
