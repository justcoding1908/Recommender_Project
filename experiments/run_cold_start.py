#!/usr/bin/env python3
"""Cold-start study: how many ratings does a new user need before personalisation beats popularity?

Usage:  python experiments/run_cold_start.py data/ml-100k results

Protocol (repeated 3 times with different random seeds)
  * Users are split into 5 groups. In turn, one group plays the "new users".
  * For each new user the rated movies are put in a random order. The LAST 10 ratings are held out as the
    test items (identical for every n). The FIRST n ratings are "revealed" to the system (n = 0, 1, 3, 5, 10).
    The user's other ratings are hidden: they are neither used for training nor counted as misses
    (they are removed from the candidate list).
  * All other users keep all their ratings. Models are retrained for every (group, n).
  * Ranking is evaluated over the full catalogue; relevant = held-out rating >= 4.
Hyper-parameters are fixed to the values most often chosen in the benchmark (PureSVD f=20, implicit
item-kNN k=100, hybrid genre weight 0.25).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.data import N_USERS, load_items, load_ratings, to_dense  # noqa: E402
from recsys.metrics import rank_eval  # noqa: E402
from recsys.models import (content_scores, itemknn_kernel, popularity, pure_svd_kernel, zrow)  # noqa: E402

N_LIST = [0, 1, 3, 5, 10]
TEST_N, REPEATS, FOLDS = 10, 3, 5
F_SVD, K_KNN, W_HYB = 20, 100, 0.25
MODELS = ["Popularity", "Content-based (genres)", "Item-kNN (implicit)", "PureSVD", "Hybrid (PureSVD + genres)"]


def main(d, out):
    os.makedirs(out, exist_ok=True)
    full = load_ratings(d)
    _, G, _ = load_items(d)
    R_full = to_dense(full)
    seen_full = R_full > 0
    runs = []  # one dict per (repeat, fold, n, model)
    t0 = time.time()
    for rep in range(REPEATS):
        rng = np.random.default_rng(1000 + rep)
        groups = np.array_split(rng.permutation(N_USERS), FOLDS)
        perm = {u: rng.permutation(np.flatnonzero(seen_full[u])) for u in range(N_USERS)}
        for g, cold in enumerate(groups):
            test = pd.DataFrame([(u, i, R_full[u, i]) for u in cold for i in perm[u][-TEST_N:]],
                                columns=["u", "i", "r"])
            exclude = np.zeros_like(seen_full)
            for u in cold:
                exclude[u, perm[u][:-TEST_N]] = True  # revealed + hidden are never candidates
            for n in N_LIST:
                R_train = R_full.copy()
                R_train[cold, :] = 0
                for u in cold:
                    rev = perm[u][:n]
                    R_train[u, rev] = R_full[u, rev]
                pop = (R_train > 0).sum(0)
                scores = {"Popularity": popularity(R_train)}
                if n > 0:  # with no ratings at all nothing can be personalised
                    svd = R_train @ pure_svd_kernel(R_train, F_SVD)
                    con = content_scores(R_train, G)
                    scores["Content-based (genres)"] = con
                    scores["Item-kNN (implicit)"] = (R_train > 0).astype(float) @ itemknn_kernel(R_train, K_KNN)
                    scores["PureSVD"] = svd
                    scores["Hybrid (PureSVD + genres)"] = zrow(svd) + W_HYB * zrow(con)
                for name, S in scores.items():
                    m = rank_eval(S, exclude, test, pop)
                    runs.append({"rep": rep, "fold": g, "n": n, "model": name, **m})
            print(f"repeat {rep + 1}/{REPEATS} fold {g + 1}/{FOLDS} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(runs)
    summary = {}
    for (model, n), grp in df.groupby(["model", "n"]):
        summary.setdefault(model, {})[int(n)] = {
            metric: {"mean": float(grp[metric].mean()), "std": float(grp[metric].std(ddof=1))}
            for metric in ["P@10", "R@10", "NDCG@10", "Coverage", "AvgPop", "Users"]}
    # paired comparison with popularity inside the same (repeat, fold, n) run
    pop_ndcg = df[df.model == "Popularity"].set_index(["rep", "fold", "n"])["NDCG@10"]
    wins = {}
    for model in MODELS[1:]:
        sub = df[df.model == model].set_index(["rep", "fold", "n"])["NDCG@10"]
        for n in N_LIST[1:]:
            idx = [k for k in sub.index if k[2] == n]
            wins.setdefault(model, {})[n] = int(sum(sub[k] > pop_ndcg[k] for k in idx))
    json.dump({"config": {"n_list": N_LIST, "test_items_per_user": TEST_N, "repeats": REPEATS, "folds": FOLDS,
                          "puresvd_f": F_SVD, "iknn_k": K_KNN, "hybrid_w": W_HYB, "runs_per_cell": REPEATS * FOLDS},
               "summary": summary, "wins_vs_popularity": wins, "runs": runs},
              open(os.path.join(out, "cold_start.json"), "w"), indent=1)
    tab = df.pivot_table(index="model", columns="n", values="NDCG@10", aggfunc="mean").reindex(MODELS)
    tab.round(4).to_csv(os.path.join(out, "cold_start_ndcg.csv"))
    print(tab.round(4).to_string())
    print("wins vs popularity (of", REPEATS * FOLDS, "runs):", json.dumps(wins))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
