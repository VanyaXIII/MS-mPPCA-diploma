"""Markov Switching mPPCA: HMM with mPPCA emissions via Baum-Welch EM.

Joint model:
  z_1 ~ Cat(π_0)
  z_t | z_{t-1} ~ Cat(A_{z_{t-1}, ·})
  x_t | z_t = k ~ N(μ_k, W_k W_k^T + σ²_k I)

E-step: HMM forward-backward using mPPCA emission log-likelihoods.
M-step: update (A, π_0) from (γ, ξ); update (μ_k, W_k, σ²_k) via mPPCA
        m_step with R := γ and update_weights=False.

The vanilla mPPCA mixing weight π_k is replaced by (π_0, A).
For downstream VaR, use A^T γ_T as the one-step-ahead mixture weight.
"""

from __future__ import annotations

import numpy as np

from src.models.hmm import forward_backward, m_step_hmm, stationary
from src.models.mppca import emission_log_likelihoods, m_step, student_e_quantities, update_nu


# ---------------------------------------------------------------------------
# Initialisation from vanilla mPPCA responsibilities
# ---------------------------------------------------------------------------

def init_ms_from_mppca(
    R: np.ndarray,
    sticky_diag: float = 0.1,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialise HMM (A, pi0) from vanilla mPPCA soft assignments R.

    Parameters
    ----------
    R           : (T, K) — mPPCA responsibilities from a converged vanilla run
    sticky_diag : extra mass on diagonal before normalisation (encourages
                  high self-transition, realistic for financial regimes)
    eps         : Dirichlet floor per cell

    Returns
    -------
    A   : (K, K)
    pi0 : (K,)
    """
    K = R.shape[1]
    # Soft co-occurrence: A_{jk} ∝ Σ_t R_{t,j} R_{t+1,k}
    A = R[:-1].T @ R[1:]
    A += sticky_diag * np.eye(K)
    A += eps
    A /= A.sum(axis=1, keepdims=True)
    pi0 = R[0] + eps
    pi0 /= pi0.sum()
    return A, pi0


# ---------------------------------------------------------------------------
# Label alignment across windows
# ---------------------------------------------------------------------------

def _best_permutation(means_new: np.ndarray, means_ref: np.ndarray) -> list[int]:
    """Find the row permutation of means_new closest to means_ref (Frobenius)."""
    from itertools import permutations
    K = means_new.shape[0]
    best_perm: list[int] = list(range(K))
    best_dist = np.inf
    for perm in permutations(range(K)):
        dist = float(np.linalg.norm(means_new[list(perm)] - means_ref))
        if dist < best_dist:
            best_dist = dist
            best_perm = list(perm)
    return best_perm


def align_labels(
    means: np.ndarray,
    W: np.ndarray,
    sigma2: np.ndarray,
    A: np.ndarray,
    pi0: np.ndarray,
    gamma: np.ndarray,
    means_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Permute cluster labels so means best match means_ref (previous window).

    Without alignment, regime k can swap identity across windows, making
    means_hist[:, k] an incoherent time-series.
    """
    perm = _best_permutation(means, means_ref)
    if perm == list(range(len(perm))):
        return means, W, sigma2, A, pi0, gamma
    p = np.array(perm)
    return (
        means[p],
        W[p],
        sigma2[p],
        A[np.ix_(p, p)],
        pi0[p],
        gamma[:, p],
    )


# ---------------------------------------------------------------------------
# One GEM iteration
# ---------------------------------------------------------------------------

def gem_step_ms(
    X: np.ndarray,
    means: np.ndarray,
    W: np.ndarray,
    sigma2: np.ndarray,
    A: np.ndarray,
    pi0: np.ndarray,
    n_components: int,
    hmm_eps: float = 1e-4,
    emission: str = "normal",
    nu: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray | None]:
    """One MS-mPPCA GEM iteration.

    E-step: HMM forward-backward with Normal or Student-t emissions → (gamma, xi).
            Student-t also produces per-sample scale weights u_{tk}.
    M-step: update (A, pi0) from (gamma, xi);
            update (mu, W, sigma2) from gamma (weighted by u for Student-t);
            update nu via ECM fixed-point (Student-t only).

    Parameters
    ----------
    emission : "normal" or "student"
    nu       : (K,) degrees of freedom per cluster; required when emission="student"

    Returns
    -------
    means, W, sigma2, A, pi0, gamma, log_lik, nu_new
    nu_new is None when emission="normal".
    """
    # E-step
    if emission == "student":
        assert nu is not None
        log_b, u = student_e_quantities(X, means, W, sigma2, nu)
    else:
        log_b = emission_log_likelihoods(X, means, W, sigma2)
        u = None

    gamma, xi, log_lik = forward_backward(
        log_b, np.log(A + 1e-300), np.log(pi0 + 1e-300)
    )

    # M-step — HMM
    A_new, pi0_new = m_step_hmm(gamma, xi, eps=hmm_eps)

    # M-step — mPPCA emissions; u=None for Normal, (T,K) for Student-t
    means_new, W_new, sigma2_new, _ = m_step(
        X, gamma, W, sigma2, n_components, update_weights=False, u=u
    )

    # M-step — degrees of freedom (Student-t only)
    if emission == "student":
        assert nu is not None and u is not None
        nu_new = update_nu(nu, u, gamma, X.shape[1])
    else:
        nu_new = None

    return means_new, W_new, sigma2_new, A_new, pi0_new, gamma, log_lik, nu_new


# ---------------------------------------------------------------------------
# EM loop
# ---------------------------------------------------------------------------

def fit_ms(
    X: np.ndarray,
    means: np.ndarray,
    W: np.ndarray,
    sigma2: np.ndarray,
    A: np.ndarray,
    pi0: np.ndarray,
    n_components: int,
    n_iter: int = 500,
    tol: float = 1e-3,
    hmm_eps: float = 1e-4,
    emission: str = "normal",
    nu: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float], np.ndarray | None]:
    """Run MS-mPPCA EM until convergence or n_iter steps.

    Parameters
    ----------
    emission : "normal" or "student"
    nu       : (K,) initial degrees of freedom; required when emission="student"

    Returns
    -------
    means, W, sigma2, A, pi0, gamma_final, llh_history, nu_final
    nu_final is None when emission="normal".
    """
    llh_history: list[float] = []
    K = len(sigma2)
    gamma_final = np.ones((X.shape[0], K)) / K
    nu_current = nu

    for _ in range(n_iter):
        prev_L = llh_history[-1] if llh_history else None
        means, W, sigma2, A, pi0, gamma_final, L, nu_current = gem_step_ms(
            X, means, W, sigma2, A, pi0, n_components,
            hmm_eps=hmm_eps, emission=emission, nu=nu_current,
        )
        llh_history.append(L)
        if prev_L is not None and abs(L - prev_L) < tol:
            break

    return means, W, sigma2, A, pi0, gamma_final, llh_history, nu_current
