"""Hidden Markov Model: log-space forward-backward and Baum-Welch M-step.

All public functions work in log-space to avoid underflow.
Exposed as standalone callables so ms_mppca.py can compose them with
the mPPCA emission log-likelihoods.
"""

from __future__ import annotations

import numpy as np
from numba import jit
from scipy.special import logsumexp


# ---------------------------------------------------------------------------
# Core forward-backward (JIT-compiled hot path)
# ---------------------------------------------------------------------------

@jit(nopython=True, cache=True, fastmath=True)
def _forward_backward_kernel(
    log_b: np.ndarray,
    log_A: np.ndarray,
    log_pi0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Numba-compiled core. Hand-rolled logsumexp over K to avoid scipy."""
    T, K = log_b.shape

    log_alpha = np.empty((T, K))
    log_beta = np.zeros((T, K))

    # Forward pass
    for k in range(K):
        log_alpha[0, k] = log_pi0[k] + log_b[0, k]

    # Reusable buffer for logsumexp over j (size K)
    buf = np.empty(K)
    for t in range(1, T):
        for k in range(K):
            # logsumexp_j(log_alpha[t-1, j] + log_A[j, k])
            mx = log_alpha[t - 1, 0] + log_A[0, k]
            buf[0] = mx
            for j in range(1, K):
                v = log_alpha[t - 1, j] + log_A[j, k]
                buf[j] = v
                if v > mx:
                    mx = v
            s = 0.0
            for j in range(K):
                s += np.exp(buf[j] - mx)
            log_alpha[t, k] = log_b[t, k] + np.log(s) + mx

    # log P(X) = logsumexp over k of log_alpha[T-1]
    mx = log_alpha[T - 1, 0]
    for k in range(1, K):
        if log_alpha[T - 1, k] > mx:
            mx = log_alpha[T - 1, k]
    s = 0.0
    for k in range(K):
        s += np.exp(log_alpha[T - 1, k] - mx)
    log_lik = np.log(s) + mx

    # Backward pass
    for k in range(K):
        log_beta[T - 1, k] = 0.0
    for t in range(T - 2, -1, -1):
        for j in range(K):
            # logsumexp_k(log_A[j, k] + log_b[t+1, k] + log_beta[t+1, k])
            mx = log_A[j, 0] + log_b[t + 1, 0] + log_beta[t + 1, 0]
            buf[0] = mx
            for k in range(1, K):
                v = log_A[j, k] + log_b[t + 1, k] + log_beta[t + 1, k]
                buf[k] = v
                if v > mx:
                    mx = v
            s = 0.0
            for k in range(K):
                s += np.exp(buf[k] - mx)
            log_beta[t, j] = np.log(s) + mx

    # gamma = normalised exp(log_alpha + log_beta)
    gamma = np.empty((T, K))
    for t in range(T):
        mx = log_alpha[t, 0] + log_beta[t, 0]
        for k in range(1, K):
            v = log_alpha[t, k] + log_beta[t, k]
            if v > mx:
                mx = v
        s = 0.0
        for k in range(K):
            gamma[t, k] = np.exp(log_alpha[t, k] + log_beta[t, k] - mx)
            s += gamma[t, k]
        for k in range(K):
            gamma[t, k] /= s

    # xi[t, j, k] = exp(log_alpha[t, j] + log_A[j, k] + log_b[t+1, k]
    #                   + log_beta[t+1, k] - log_lik)
    xi = np.empty((T - 1, K, K))
    for t in range(T - 1):
        for j in range(K):
            for k in range(K):
                xi[t, j, k] = np.exp(
                    log_alpha[t, j]
                    + log_A[j, k]
                    + log_b[t + 1, k]
                    + log_beta[t + 1, k]
                    - log_lik
                )

    return gamma, xi, log_lik


def forward_backward(
    log_b: np.ndarray,
    log_A: np.ndarray,
    log_pi0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Log-space Baum-Welch E-step.

    Parameters
    ----------
    log_b   : (T, K) — per-state emission log-likelihoods
    log_A   : (K, K) — log transition matrix (rows sum to 1 in prob space)
    log_pi0 : (K,)   — log initial distribution

    Returns
    -------
    gamma   : (T, K)       — smoothed state posteriors, rows sum to 1
    xi      : (T-1, K, K)  — pairwise posteriors; xi[t, j, k] = P(z_t=j, z_{t+1}=k | X)
    log_lik : float        — log P(X | params)
    """
    # Ensure contiguous float64 arrays for the numba kernel
    log_b_c = np.ascontiguousarray(log_b, dtype=np.float64)
    log_A_c = np.ascontiguousarray(log_A, dtype=np.float64)
    log_pi0_c = np.ascontiguousarray(log_pi0, dtype=np.float64)
    gamma, xi, log_lik = _forward_backward_kernel(log_b_c, log_A_c, log_pi0_c)
    return gamma, xi, float(log_lik)


# ---------------------------------------------------------------------------
# HMM M-step
# ---------------------------------------------------------------------------

def m_step_hmm(
    gamma: np.ndarray,
    xi: np.ndarray,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Update transition matrix A and initial distribution pi0.

    Parameters
    ----------
    gamma : (T, K)
    xi    : (T-1, K, K)
    eps   : Dirichlet smoothing per cell (prevents zero rows when a regime is rare)

    Returns
    -------
    A   : (K, K) rows sum to 1
    pi0 : (K,)   sums to 1
    """
    A_counts = xi.sum(axis=0) + eps          # (K, K)
    A = A_counts / A_counts.sum(axis=1, keepdims=True)
    pi0 = gamma[0] + eps
    pi0 /= pi0.sum()
    return A, pi0


# ---------------------------------------------------------------------------
# Stationary distribution
# ---------------------------------------------------------------------------

def stationary(A: np.ndarray) -> np.ndarray:
    """Left eigenvector of A for eigenvalue 1 (long-run regime probabilities)."""
    eigvals, eigvecs = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    stat = np.real(eigvecs[:, idx])
    stat = np.abs(stat)
    return stat / stat.sum()


# ---------------------------------------------------------------------------
# Simple univariate Gaussian HMM (for demonstration / notebook)
# ---------------------------------------------------------------------------

def _gaussian_log_b(X: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """(T,) observations, (K,) means/variances -> (T, K) log-likelihoods."""
    T = len(X)
    K = len(mu)
    log_b = np.empty((T, K))
    for k in range(K):
        diff = X - mu[k]
        log_b[:, k] = (
            -0.5 * np.log(2.0 * np.pi * sigma2[k])
            - 0.5 * diff ** 2 / sigma2[k]
        )
    return log_b


def fit_gaussian_hmm(
    X: np.ndarray,
    K: int = 2,
    n_iter: int = 200,
    tol: float = 1e-4,
    random_state: int = 42,
) -> dict:
    """Fit a K-state HMM with univariate Gaussian emissions via Baum-Welch.

    Parameters
    ----------
    X     : (T,) 1-D observation sequence
    K     : number of hidden states
    n_iter, tol : convergence controls

    Returns
    -------
    dict with keys: mu, sigma2, A, pi0, gamma, xi, llh_history
    """
    T = len(X)
    rng = np.random.default_rng(random_state)

    # Initialise: K-means style split
    sorted_idx = np.argsort(X)
    mu = np.array([X[sorted_idx[int(T * (k + 0.5) / K)]].item() for k in range(K)])
    sigma2 = np.full(K, X.var())
    A = (np.ones((K, K)) + rng.uniform(0, 0.1, (K, K)))
    A /= A.sum(axis=1, keepdims=True)
    pi0 = np.ones(K) / K

    llh_history: list[float] = []
    gamma_final = np.ones((T, K)) / K
    xi_final = np.ones((T - 1, K, K)) / K**2

    for _ in range(n_iter):
        # E-step
        log_b = _gaussian_log_b(X, mu, sigma2)
        gamma, xi, log_lik = forward_backward(log_b, np.log(A + 1e-300), np.log(pi0 + 1e-300))
        llh_history.append(log_lik)
        gamma_final, xi_final = gamma, xi

        # M-step — HMM params
        A, pi0 = m_step_hmm(gamma, xi)

        # M-step — Gaussian emission params
        N_k = gamma.sum(axis=0)            # (K,)
        mu = (gamma * X[:, None]).sum(axis=0) / N_k
        diff2 = (X[:, None] - mu[None, :]) ** 2
        sigma2 = (gamma * diff2).sum(axis=0) / N_k
        sigma2 = np.maximum(sigma2, 1e-8)

        if len(llh_history) > 1 and abs(llh_history[-1] - llh_history[-2]) < tol:
            break

    return {
        "mu": mu,
        "sigma2": sigma2,
        "A": A,
        "pi0": pi0,
        "gamma": gamma_final,
        "xi": xi_final,
        "llh_history": llh_history,
    }
