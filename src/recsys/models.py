"""Recommendation models. All take a dense user x item rating matrix R (0 = not rated).

Rating-prediction models return a matrix of predicted ratings (unclipped).
Ranking models return a matrix of scores (higher = better); only the order matters.
"""
from __future__ import annotations

import numpy as np

REL = 4.0  # a rating >= REL counts as "liked"


# ------------------------------------------------------------ rating prediction
def fit_bias(R, reg_i=10.0, reg_u=15.0, iters=10):
    """Regularised baseline: mu + user bias + item bias."""
    nu_, ni_ = R.shape
    M = R > 0
    mu = R[M].mean()
    bu, bi = np.zeros(nu_), np.zeros(ni_)
    nu, ni = M.sum(1), M.sum(0)
    for _ in range(iters):
        bi = ((R - mu - bu[:, None]) * M).sum(0) / (reg_i + ni)
        bu = ((R - mu - bi[None, :]) * M).sum(1) / (reg_u + nu)
    return mu + bu[:, None] + bi[None, :]


def cosine(X):
    """Cosine similarity between the rows of X."""
    n = np.linalg.norm(X, axis=1)
    n[n == 0] = 1
    return (X @ X.T) / np.outer(n, n)


def topk_rows(S, k):
    """Keep the k largest positive entries of each row (diagonal removed), zero the rest."""
    S = S.copy()
    np.fill_diagonal(S, 0)
    S[S < 0] = 0
    if k < S.shape[1]:
        drop = np.argpartition(-S, k, axis=1)[:, k:]
        np.put_along_axis(S, drop, 0, axis=1)
    return S


def knn_predict(R, Pb, k, mode, shrink=10.0, damp=0.5):
    """kNN on baseline residuals; similarity is shrunk by the number of co-rated items."""
    M = (R > 0).astype(float)
    E = (R - Pb) * M
    if mode == "user":
        S = cosine(E)
        C = M @ M.T
        Sk = topk_rows(S * C / (C + shrink), k)
        return Pb + (Sk @ E) / ((Sk @ M) + damp)
    S = cosine(E.T)
    C = M.T @ M
    Sk = topk_rows(S * C / (C + shrink), k)
    return Pb + (E @ Sk.T) / ((M @ Sk.T) + damp)


def als_mf(R, Pb, f, lam, iters=12, seed=0):
    """Regularised matrix factorisation on baseline residuals (ALS with weighted-lambda)."""
    nu_, ni_ = R.shape
    rng = np.random.default_rng(seed)
    M = R > 0
    E = (R - Pb) * M
    P = rng.normal(0, 0.1, (nu_, f))
    Q = rng.normal(0, 0.1, (ni_, f))
    I = np.eye(f)
    ui = [np.flatnonzero(M[u]) for u in range(nu_)]
    iu = [np.flatnonzero(M[:, i]) for i in range(ni_)]
    for _ in range(iters):
        for u in range(nu_):
            idx = ui[u]
            if len(idx):
                Qi = Q[idx]
                P[u] = np.linalg.solve(Qi.T @ Qi + lam * len(idx) * I, Qi.T @ E[u, idx])
        for i in range(ni_):
            idx = iu[i]
            if len(idx):
                Pu = P[idx]
                Q[i] = np.linalg.solve(Pu.T @ Pu + lam * len(idx) * I, Pu.T @ E[idx, i])
    Q[M.sum(0) == 0] = 0
    return Pb + P @ Q.T


# ------------------------------------------------------------ ranking models
def pure_svd(R, f):
    """PureSVD: rank-f truncated SVD of R with missing entries as 0 (Cremonesi et al., 2010)."""
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    return (U[:, :f] * s[:f]) @ Vt[:f]


def pure_svd_kernel(R, f):
    """K = V_f V_f^T. A rating row x is scored by x @ K; for a training row this equals pure_svd."""
    _, _, Vt = np.linalg.svd(R, full_matrices=False)
    V = Vt[:f].T
    return V @ V.T


def itemknn_implicit(R, k):
    B = (R > 0).astype(float)
    return B @ topk_rows(cosine(B.T), k).T


def itemknn_kernel(R, k):
    """K[j, i] = similarity of item i to its neighbour j; a binary row x is scored by x @ K."""
    B = (R > 0).astype(float)
    return topk_rows(cosine(B.T), k).T


def popularity(R):
    return np.tile((R > 0).sum(0).astype(float), (R.shape[0], 1))


def content_scores(R, G):
    """Genre-based content filtering: IDF-weighted genre profile of liked items, cosine to each item."""
    ni_ = G.shape[0]
    idf = np.log(ni_ / (1.0 + G.sum(0)))
    X = G * idf
    n = np.linalg.norm(X, axis=1)
    n[n == 0] = 1
    prof = (R >= REL).astype(float) @ X
    pn = np.linalg.norm(prof, axis=1)
    pn[pn == 0] = 1
    return (prof / pn[:, None]) @ (X / n[:, None]).T


def zrow(S):
    """Standardise each row (user) to mean 0, std 1 so scores from different models can be blended."""
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
