"""Mixture of Probabilistic PCA (mPPCA) via Generalized EM.

Design note for MS coupling
---------------------------
`e_step` and `m_step` are exposed as standalone callables so the joint
MS-mPPCA EM can swap them:
  - MS supplies γ_t (forward-backward posteriors) in place of mPPCA responsibilities
    → pass those as `responsibilities` directly into `m_step`.
  - mPPCA supplies R_tn (soft cluster assignments) as the E distribution
    → pass R into the MS M-step (transition matrix update).
The `update_weights` flag in `m_step` lets the MS layer own π_k updates.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma, gammaln
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.utils.numba_utils import logsumexp_rows


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_kmeans_pca(
    X: np.ndarray,
    n_components: int,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialise mPPCA parameters via K-Means + per-cluster PCA.

    Returns
    -------
    means   : (K, D)
    W       : (K, D, q)
    sigma2  : (K,)
    weights : (K,)
    """
    n_samples, n_features = X.shape
    min_size = max(n_components + 1, int(n_samples / n_clusters / 3) + 1)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)

    means = np.zeros((n_clusters, n_features))
    W = np.zeros((n_clusters, n_features, n_components))
    sigma2 = np.zeros(n_clusters)
    weights = np.zeros(n_clusters)

    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if len(idx) < min_size:
            extra = np.random.choice(
                np.setdiff1d(np.arange(n_samples), idx),
                size=min_size - len(idx),
                replace=False,
            )
            idx = np.concatenate([idx, extra])
            labels[extra] = k

        data_k = X[idx]
        pca = PCA(n_components=n_components)
        pca.fit(data_k)

        means[k] = data_k.mean(axis=0)
        W[k] = pca.components_.T
        sigma2[k] = max(pca.noise_variance_, 1e-4)
        weights[k] = len(idx) / n_samples

    weights /= weights.sum()
    return means, W, sigma2, weights


# ---------------------------------------------------------------------------
# Shared Woodbury computation (used by both Normal and Student-t emissions)
# ---------------------------------------------------------------------------

def _compute_mahal_logdet(
    X: np.ndarray,
    means: np.ndarray,
    W: np.ndarray,
    sigma2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mahalanobis distances and log-determinants via Woodbury identity.

    Returns
    -------
    mahal     : (T, K)  δ_{tk} = (x_t - μ_k)^T C_k^{-1} (x_t - μ_k)
    log_det_C : (K,)    log|C_k| = (D-q) log σ²_k + log|M_k|
    """
    T, D = X.shape
    K = len(sigma2)
    q = W[0].shape[1]
    reg = 1e-4
    eye_q = np.eye(q)

    mahal = np.empty((T, K))
    log_det_C = np.empty(K)

    for k in range(K):
        s2 = sigma2[k]
        diff = X - means[k]
        Wk = W[k]
        M = Wk.T @ Wk + (s2 + reg) * eye_q
        M_inv = np.linalg.inv(M)

        proj = diff @ Wk
        sq_diff = np.einsum("td,td->t", diff, diff)
        quad = np.einsum("tq,qr,tr->t", proj, M_inv, proj)
        mahal[:, k] = (sq_diff - quad) / s2

        _, log_det_M = np.linalg.slogdet(M)
        log_det_C[k] = (D - q) * np.log(s2) + log_det_M

    return mahal, log_det_C


# ---------------------------------------------------------------------------
# Normal emission log-likelihoods (shared by e_step and HMM forward-backward)
# ---------------------------------------------------------------------------

def emission_log_likelihoods(
    X: np.ndarray,
    means: np.ndarray,
    W: np.ndarray,
    sigma2: np.ndarray,
) -> np.ndarray:
    """Per-cluster log N(x_t | μ_k, C_k) without mixing weights.

    Parameters
    ----------
    X      : (T, D)
    means  : (K, D)
    W      : (K, D, q)
    sigma2 : (K,)

    Returns
    -------
    log_b : (T, K)
    """
    D = X.shape[1]
    const = -0.5 * D * np.log(2.0 * np.pi)
    mahal, log_det_C = _compute_mahal_logdet(X, means, W, sigma2)
    return const - 0.5 * mahal - 0.5 * log_det_C[np.newaxis, :]


# ---------------------------------------------------------------------------
# Student-t emission utilities
# ---------------------------------------------------------------------------

def student_e_quantities(
    X: np.ndarray,
    means: np.ndarray,
    W: np.ndarray,
    sigma2: np.ndarray,
    nu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster log t_ν(x_t | μ_k, C_k) and E-step scale weights u_{tk}.

    Both quantities share the same Mahalanobis computation, so they are
    produced in one pass to avoid redundant work.

    Parameters
    ----------
    X      : (T, D)
    means  : (K, D)
    W      : (K, D, q)
    sigma2 : (K,)
    nu     : (K,)  — degrees of freedom per cluster (ν > 2 required)

    Returns
    -------
    log_b : (T, K)  — log t_ν_k(x_t | μ_k, C_k), no mixing weights
    u     : (T, K)  — E[precision weight] = (ν_k + D) / (ν_k + δ_{tk})
    """
    D = X.shape[1]
    K = len(sigma2)
    mahal, log_det_C = _compute_mahal_logdet(X, means, W, sigma2)

    log_b = np.empty_like(mahal)
    u = np.empty_like(mahal)

    for k in range(K):
        nu_k = nu[k]
        log_norm = (
            gammaln(0.5 * (nu_k + D))
            - gammaln(0.5 * nu_k)
            - 0.5 * D * np.log(nu_k * np.pi)
            - 0.5 * log_det_C[k]
        )
        log_b[:, k] = log_norm - 0.5 * (nu_k + D) * np.log1p(mahal[:, k] / nu_k)
        u[:, k] = (nu_k + D) / (nu_k + mahal[:, k])

    return log_b, u


def update_nu(
    nu: np.ndarray,
    u: np.ndarray,
    R: np.ndarray,
    D: int,
    nu_min: float = 2.1,
    nu_max: float = 50.0,
) -> np.ndarray:
    """M-step for degrees of freedom ν_k via Brent root-finding.

    Solves per cluster: -ψ(ν/2) + log(ν/2) + 1 + c_k = 0
    where c_k = (1/N_k) Σ_t r_{tk} [log u_{tk} − u_{tk}].

    Parameters
    ----------
    nu    : (K,)   current degrees of freedom
    u     : (T, K) scale weights from student_e_quantities
    R     : (T, K) responsibilities
    D     : observation dimension
    """
    K = len(nu)
    nu_new = nu.copy()
    N_k = R.sum(axis=0)

    for k in range(K):
        nk = float(N_k[k])
        if nk < 1e-6:
            continue
        rk = R[:, k]
        uk = np.maximum(u[:, k], 1e-300)
        c = float((rk @ (np.log(uk) - uk)) / nk)

        def eq(v: float) -> float:
            return -digamma(0.5 * v) + np.log(0.5 * v) + 1.0 + c

        try:
            if eq(nu_min) * eq(nu_max) < 0:
                nu_new[k] = brentq(eq, nu_min, nu_max, maxiter=50)
            elif eq(nu_max) > 0:
                nu_new[k] = nu_max   # Gaussian limit
            # else: keep current nu[k] (rare degenerate case)
        except Exception:
            nu_new[k] = nu[k]

    return nu_new


# ---------------------------------------------------------------------------
# E-step
# ---------------------------------------------------------------------------

def e_step(
    X: np.ndarray,
    means: np.ndarray,
    W: np.ndarray,
    sigma2: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute soft responsibilities and log-likelihood.

    Parameters
    ----------
    X       : (T, D)
    means   : (K, D)
    W       : (K, D, q)
    sigma2  : (K,)
    weights : (K,)

    Returns
    -------
    R : (T, K)  — soft cluster responsibilities (rows sum to 1)
    L : float   — observed-data log-likelihood
    """
    log_b = emission_log_likelihoods(X, means, W, sigma2)
    log_R = log_b + np.log(weights)[None, :]
    log_sum = logsumexp_rows(log_R)
    R = np.exp(log_R - log_sum[:, None])
    L = float(log_sum.sum())
    return R, L


# ---------------------------------------------------------------------------
# M-step
# ---------------------------------------------------------------------------

def m_step(
    X: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    sigma2: np.ndarray,
    n_components: int,
    update_weights: bool = True,
    u: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update mPPCA parameters given responsibilities R.

    Parameters
    ----------
    X            : (T, D)
    R            : (T, K) — responsibilities (from E-step or MS forward-backward)
    W            : (K, D, q) — current factor loadings
    sigma2       : (K,)
    n_components : q
    update_weights : if False, keep weights fixed (MS owns π_k).
    u            : (T, K) optional Student-t scale weights; if given, μ/W/σ²
                   updates use R * u as effective weights while mixing weights
                   still use R alone.

    Returns
    -------
    means_new   : (K, D)
    W_new       : (K, D, q)
    sigma2_new  : (K,)
    weights_new : (K,)
    """
    T, D = X.shape
    K = R.shape[1]
    reg = 1e-4
    eye_q = np.eye(n_components)

    # Student-t: effective weights for sufficient statistics are R * u.
    # Mixing weights π_k still use raw R (E[z_{tk}] marginally).
    R_eff = R * u if u is not None else R

    means_new = np.zeros((K, D))
    W_new = W.copy()
    sigma2_new = sigma2.copy()
    N_k_all    = R.sum(axis=0)      # (K,) — for collapse check and weights
    Neff_k_all = R_eff.sum(axis=0)  # (K,) — effective count for stat updates
    weights_new = N_k_all / T

    _min_cluster_size = 7

    for k in range(K):
        N_k    = N_k_all[k]
        Neff_k = Neff_k_all[k]
        resp_eff = R_eff[:, k]

        if N_k < _min_cluster_size:
            _reinitialise_cluster(k, X, W_new, sigma2_new, n_components)
            means_new[k] = X.mean(axis=0)
            weights_new[k] = 1.0 / K
            continue

        means_new[k] = (resp_eff @ X) / Neff_k

        diff = X - means_new[k]
        Wk = W[k]
        s2 = sigma2[k]
        M = Wk.T @ Wk + (s2 + reg) * eye_q
        M_inv = np.linalg.inv(M)

        diff_W = diff @ Wk
        weighted_diff = diff * resp_eff[:, None]
        SW = (weighted_diff.T @ diff_W) / Neff_k

        Blank = s2 * eye_q + M_inv @ (Wk.T @ SW)
        W_new_k = SW @ np.linalg.inv(Blank)
        W_new[k] = W_new_k

        sigma2_raw = np.einsum("td,td->", weighted_diff, diff) / Neff_k
        sigma2_raw -= np.einsum("dq,qr,dr->", SW, M_inv, W_new_k)
        sigma2_raw /= D
        sigma2_new[k] = max(sigma2_raw, 1e-6)

    if not update_weights:
        weights_new = N_k_all / T

    weights_new = np.clip(weights_new, 1e-8, None)
    weights_new /= weights_new.sum()
    return means_new, W_new, sigma2_new, weights_new


def _reinitialise_cluster(
    k: int,
    X: np.ndarray,
    W: np.ndarray,
    sigma2: np.ndarray,
    n_components: int,
    subset_size: int = 50,
) -> None:
    """Replace a collapsed cluster with a random PCA-based reinitialisation (in-place)."""
    T = X.shape[0]
    idx = np.random.choice(T, size=min(subset_size, T), replace=False)
    data_sub = X[idx]
    U, S, Vt = np.linalg.svd(data_sub - data_sub.mean(axis=0), full_matrices=False)
    q = min(n_components, Vt.shape[0])
    W_init = Vt[:q].T
    if q < n_components:
        pad = np.zeros((X.shape[1], n_components - q))
        W_init = np.concatenate([W_init, pad], axis=1)
    W[k] = W_init
    leftover = max(0.0, np.sum(S**2) / T - np.sum(S[:q] ** 2) / T)
    sigma2[k] = max(leftover / max(1, X.shape[1] - q), 1e-4)


# ---------------------------------------------------------------------------
# GEM (combined one-pass E+M)
# ---------------------------------------------------------------------------

def gem_step(
    X: np.ndarray,
    means: np.ndarray,
    W: np.ndarray,
    sigma2: np.ndarray,
    weights: np.ndarray,
    n_components: int,
    update_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """One GEM iteration: E-step then M-step.

    Returns updated (means, W, sigma2, weights, R, L).
    R is returned so the MS layer can consume it without re-running E-step.
    """
    R, L = e_step(X, means, W, sigma2, weights)
    means_new, W_new, sigma2_new, weights_new = m_step(
        X, R, W, sigma2, n_components, update_weights=update_weights
    )
    return means_new, W_new, sigma2_new, weights_new, R, L


# ---------------------------------------------------------------------------
# EM loop
# ---------------------------------------------------------------------------

def fit(
    X: np.ndarray,
    means: np.ndarray,
    W: np.ndarray,
    sigma2: np.ndarray,
    weights: np.ndarray,
    n_components: int,
    n_iter: int = 2000,
    tol: float = 1e-3,
    update_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Run EM until convergence or ``n_iter`` iterations.

    Returns
    -------
    means, W, sigma2, weights, R_final, llh_history
    """
    llh_history: list[float] = []
    R_final = np.ones((X.shape[0], len(sigma2))) / len(sigma2)

    for it in range(n_iter):
        prev_L = llh_history[-1] if llh_history else None
        means, W, sigma2, weights, R_final, L = gem_step(
            X, means, W, sigma2, weights, n_components, update_weights
        )
        llh_history.append(L)

        if prev_L is not None and abs(L - prev_L) < tol:
            break

    return means, W, sigma2, weights, R_final, llh_history
