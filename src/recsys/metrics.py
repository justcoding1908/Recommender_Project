"""Evaluation metrics: rating error and top-N ranking quality."""
from __future__ import annotations

import numpy as np

REL, TOPN = 4.0, 10


def rating_eval(P, test):
    """RMSE and MAE of predictions clipped to [1, 5] on the (u, i, r) rows of `test`."""
    pred = np.clip(P[test["u"].values, test["i"].values], 1, 5)
    err = pred - test["r"].values
    return {"RMSE": float(np.sqrt((err ** 2).mean())), "MAE": float(np.abs(err).mean())}


def rank_eval(scores, exclude, test, item_pop, topn=TOPN):
    """Full-catalogue ranking evaluation.

    scores  : (users x items) score matrix, higher = better
    exclude : boolean (users x items) matrix of items that must not be recommended
              (training items, plus any other known items that are not in `test`)
    test    : DataFrame with columns u, i, r; an item is relevant if r >= 4
    item_pop: number of training ratings per item (for the average-popularity metric)

    Only users with at least one relevant test item are evaluated.
    """
    n_items = scores.shape[1]
    rel = test[test["r"] >= REL]
    rel_sets = {u: set(g["i"].values) for u, g in rel.groupby("u")}
    S = scores - np.arange(n_items)[None, :] * 1e-9  # deterministic tie-break by item id
    S[exclude] = -np.inf
    top = np.argpartition(-S, topn, axis=1)[:, :topn]
    P, Rc, N, recs = [], [], [], []
    disc = 1.0 / np.log2(np.arange(topn) + 2)
    for u, rs in rel_sets.items():
        items = top[u][np.argsort(-S[u, top[u]])]
        hit = np.array([it in rs for it in items], dtype=float)
        P.append(hit.sum() / topn)
        Rc.append(hit.sum() / len(rs))
        N.append((hit * disc).sum() / disc[: min(len(rs), topn)].sum())
        recs.append(items)
    recs = np.concatenate(recs)
    return {
        "P@10": float(np.mean(P)), "R@10": float(np.mean(Rc)), "NDCG@10": float(np.mean(N)),
        "Coverage": float(len(np.unique(recs)) / n_items),
        "AvgPop": float(item_pop[recs].mean()),
        "Users": len(P),
    }
