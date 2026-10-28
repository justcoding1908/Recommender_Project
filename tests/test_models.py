"""Unit tests on small synthetic data (no dataset needed)."""
import numpy as np
import pandas as pd
import pytest

from recsys.metrics import rank_eval
from recsys.models import (fit_bias, itemknn_implicit, itemknn_kernel, pure_svd, pure_svd_kernel, zrow)
from recsys.recommend import Recommender


@pytest.fixture
def R():
    rng = np.random.default_rng(0)
    M = rng.random((40, 60)) < 0.25
    return M * rng.integers(1, 6, (40, 60)).astype(float)


def test_pure_svd_kernel_reproduces_training_rows(R):
    # Fold-in math: scoring a training row x with x @ K must equal the PureSVD reconstruction of that row.
    assert np.allclose(R @ pure_svd_kernel(R, 6), pure_svd(R, 6), atol=1e-8)


def test_itemknn_kernel_reproduces_implicit_scores(R):
    B = (R > 0).astype(float)
    assert np.allclose(B @ itemknn_kernel(R, 10), itemknn_implicit(R, 10))


def test_bias_baseline_shape_and_mean(R):
    P = fit_bias(R)
    assert P.shape == R.shape
    assert abs(P[R > 0].mean() - R[R > 0].mean()) < 0.15


def test_zrow_is_standardised():
    Z = zrow(np.random.default_rng(1).normal(3, 2, (5, 50)))
    assert np.allclose(Z.mean(1), 0, atol=1e-9) and np.allclose(Z.std(1), 1, atol=1e-6)


def test_rank_eval_perfect_ranking_has_ndcg_one():
    n_users, n_items = 4, 30
    rng = np.random.default_rng(2)
    rel = {u: rng.choice(n_items, 3, replace=False) for u in range(n_users)}
    test = pd.DataFrame([(u, i, 5.0) for u, items in rel.items() for i in items], columns=["u", "i", "r"])
    scores = np.zeros((n_users, n_items))
    for u, items in rel.items():
        scores[u, items] = 10
    m = rank_eval(scores, np.zeros((n_users, n_items), bool), test, np.ones(n_items))
    assert m["NDCG@10"] == pytest.approx(1.0) and m["R@10"] == pytest.approx(1.0)
    assert m["P@10"] == pytest.approx(0.3)


def test_rank_eval_respects_exclusions():
    scores = np.array([[9.0, 8.0, 1.0, 0.5, 0.2, 0.1, 0.0, -1, -2, -3, -4, -5]])
    test = pd.DataFrame([(0, 0, 5.0)], columns=["u", "i", "r"])
    exclude = np.zeros_like(scores, dtype=bool)
    exclude[0, 0] = True  # the relevant item is excluded, so it can never be a hit
    assert rank_eval(scores, exclude, test, np.ones(12))["P@10"] == 0.0


def test_recommender_scores_are_sum_of_contributions(R):
    n_items = R.shape[1]
    G = (np.random.default_rng(3).random((n_items, 5)) < 0.3).astype(float)
    items = pd.DataFrame({"title": [f"m{i}" for i in range(n_items)], "year": ["2000"] * n_items,
                          "genres": [["g"]] * n_items})
    rec = Recommender(R, G, items, f=5, k=10)
    liked = {3: 5.0, 10: 4.0, 20: 5.0}
    s = rec.scores(liked, "puresvd")
    for i in (0, 7, 30):
        assert s[i] == pytest.approx(sum(rec.contributions(liked, "puresvd", i).values()))
    out = rec.recommend(liked, "hybrid", n=10)
    assert len(out) == 10 and not ({o["movie_id"] - 1 for o in out} & set(liked))
