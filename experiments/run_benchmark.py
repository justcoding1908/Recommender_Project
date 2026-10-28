#!/usr/bin/env python3
"""Benchmark of ten recommenders on MovieLens 100K (official 5-fold splits u1..u5).

Usage:  python experiments/run_benchmark.py data/ml-100k results

Rating prediction : global mean, bias baseline, user-kNN, item-kNN, ALS matrix factorisation
Top-N ranking     : the same models (predicted rating as score) + popularity, implicit item-kNN,
                    PureSVD, genre content-based, PureSVD + genres hybrid
Protocol: hyper-parameters tuned per fold on a 10% hold-out of that fold's training data,
relevant item = test rating >= 4, full-catalogue ranking (all items not in the user's training set).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.data import N_ITEMS, N_USERS, load_items, load_ratings, to_dense  # noqa: E402
from recsys.metrics import rank_eval, rating_eval  # noqa: E402
from recsys.models import (als_mf, content_scores, fit_bias, itemknn_implicit, knn_predict,  # noqa: E402
                           popularity, pure_svd, zrow)

SEED = 42


def tune(train, G, log):
    rng = np.random.default_rng(SEED)
    msk = rng.random(len(train)) < 0.10
    val, tr = train[msk], train[~msk]
    R = to_dense(tr)
    Pb = fit_bias(R)
    pop = (R > 0).sum(0)
    best = {}
    for mode in ("user", "item"):
        res = {k: rating_eval(knn_predict(R, Pb, k, mode), val)["RMSE"] for k in (10, 20, 40, 80, 160)}
        best[f"{mode}_knn_k"] = min(res, key=res.get)
        log(f"tune {mode}-kNN RMSE by k: {res}")
    res = {}
    for f in (5, 10, 20, 40):
        for lam in (0.02, 0.05, 0.1, 0.2):
            res[(f, lam)] = rating_eval(als_mf(R, Pb, f, lam, iters=10), val)["RMSE"]
    (bf, bl) = min(res, key=res.get)
    best["als_f"], best["als_lam"] = bf, bl
    log(f"tune ALS RMSE: {({str(k): round(v, 4) for k, v in res.items()})}")
    res = {f: rank_eval(pure_svd(R, f), R > 0, val, pop)["NDCG@10"] for f in (10, 20, 40, 80)}
    best["puresvd_f"] = max(res, key=res.get)
    log(f"tune PureSVD NDCG@10 by f: {res}")
    Zs, Zc = zrow(pure_svd(R, best["puresvd_f"])), zrow(content_scores(R, G))
    res = {w: rank_eval(Zs + w * Zc, R > 0, val, pop)["NDCG@10"] for w in (0.0, 0.1, 0.25, 0.5, 1.0)}
    best["hybrid_w"] = max(res, key=res.get)
    log(f"tune hybrid weight on genres, NDCG@10: {res}")
    res = {k: rank_eval(itemknn_implicit(R, k), R > 0, val, pop)["NDCG@10"] for k in (20, 50, 100, 200)}
    best["iknn_impl_k"] = max(res, key=res.get)
    log(f"tune implicit item-kNN NDCG@10 by k: {res}")
    return best


def main(d, out):
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    full = load_ratings(d, "u.data")
    M = np.zeros((N_USERS, N_ITEMS), bool)
    M[full["u"], full["i"]] = True
    cnt_i = M.sum(0)
    srt = np.sort(cnt_i)[::-1]
    stats = {
        "users": int(full["u"].nunique()), "items": int(full["i"].nunique()), "ratings": int(len(full)),
        "sparsity": float(1 - len(full) / (N_USERS * N_ITEMS)),
        "mean_rating": float(full["r"].mean()),
        "rating_dist": {int(k): int(v) for k, v in full["r"].value_counts().sort_index().items()},
        "ratings_per_user_mean": float(M.sum(1).mean()), "ratings_per_user_median": float(np.median(M.sum(1))),
        "ratings_per_user_min": int(M.sum(1).min()), "ratings_per_user_max": int(M.sum(1).max()),
        "ratings_per_item_median": float(np.median(cnt_i)),
        "top20pct_items_share": float(srt[: int(0.2 * N_ITEMS)].sum() / srt.sum()),
        "items_with_le5_ratings": float((cnt_i <= 5).mean()),
        "pct_rating_ge4": float((full["r"] >= 4).mean()),
    }
    log("dataset stats: " + json.dumps(stats))
    np.save(os.path.join(out, "item_counts.npy"), cnt_i)

    _, G, _ = load_items(d)
    folds, all_params = [], []
    for f in range(1, 6):
        train, test = load_ratings(d, f"u{f}.base"), load_ratings(d, f"u{f}.test")
        t0 = time.time()
        params = tune(train, G, log)  # tuned only on a 10% hold-out of this fold's own training data
        all_params.append({k: (float(v) if isinstance(v, float) else int(v)) for k, v in params.items()})
        log(f"fold {f} chosen params: {params}  (tuning {time.time() - t0:.0f}s)")
        R = to_dense(train)
        seen = R > 0
        pop = seen.sum(0)
        Pb = fit_bias(R)
        mu = R[R > 0].mean()
        ts = time.time()
        explicit = {
            "Global mean": np.full(R.shape, mu),
            "Bias baseline": Pb,
            "User-kNN": knn_predict(R, Pb, params["user_knn_k"], "user"),
            "Item-kNN": knn_predict(R, Pb, params["item_knn_k"], "item"),
            "Matrix factorisation (ALS)": als_mf(R, Pb, params["als_f"], params["als_lam"]),
        }
        Zs, Zc = zrow(pure_svd(R, params["puresvd_f"])), zrow(content_scores(R, G))
        ranking_only = {
            "Popularity": popularity(R),
            "Content-based (genres)": content_scores(R, G),
            "Hybrid (PureSVD + genres)": Zs + params["hybrid_w"] * Zc,
            "Item-kNN (implicit)": itemknn_implicit(R, params["iknn_impl_k"]),
            "PureSVD": pure_svd(R, params["puresvd_f"]),
        }
        res = {}
        for name, P in explicit.items():
            res[name] = {}
            if name != "Global mean":
                res[name].update(rank_eval(P, seen, test, pop))
            res[name].update(rating_eval(P, test))
        for name, P in ranking_only.items():
            res[name] = rank_eval(P, seen, test, pop)
        folds.append(res)
        log(f"fold {f} done in {time.time() - ts:.0f}s: " + json.dumps(
            {k: {m: round(v, 4) for m, v in r.items() if m in ('RMSE', 'NDCG@10')} for k, r in res.items()}))

    models = list(folds[0].keys())
    summary = {}
    for m in models:
        summary[m] = {}
        for metric in folds[0][m]:
            vals = np.array([fo[m][metric] for fo in folds])
            summary[m][metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    json.dump({"stats": stats, "params_per_fold": all_params, "folds": folds, "summary": summary},
              open(os.path.join(out, "results.json"), "w"), indent=2)
    rows = []
    for m in models:
        row = {"Model": m}
        for metric, v in summary[m].items():
            row[metric] = f"{v['mean']:.4f} ± {v['std']:.4f}" if metric not in ("Users", "AvgPop") else f"{v['mean']:.1f}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, "summary.csv"), index=False)
    log("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
