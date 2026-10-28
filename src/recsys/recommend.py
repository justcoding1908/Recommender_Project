"""Recommend for a brand-new user from a handful of liked movies ("fold-in"), with explanations.

PureSVD and implicit item-kNN are both *linear* item-item models: a user's score for item i is
    score(i) = sum_j x_j * K[j, i]
where x holds the user's ratings (or 0/1 flags) and K is an item-item kernel
(K = V_f V_f^T for PureSVD, the neighbour-similarity matrix for item-kNN).
Because the score is a sum over the user's movies, each movie's share of a recommendation
can be read off directly, which gives a faithful "because you liked ..." explanation.
"""
from __future__ import annotations

import numpy as np

from .models import REL, content_scores, itemknn_kernel, pure_svd_kernel, zrow

MODELS = {
    "hybrid": "PureSVD blended with a genre profile (best accuracy in the benchmark)",
    "puresvd": "PureSVD: low-rank collaborative filtering",
    "item_knn": "Implicit item-kNN: movies rated by the same people as yours",
    "content": "Content-based: movies whose genres match your liked movies",
    "popularity": "Popularity: most-rated movies overall (no personalisation)",
}


class Recommender:
    def __init__(self, R, G, items, f=20, k=100, w=0.25):
        self.R, self.G, self.items, self.w = R, G, items, w
        self.n_items = R.shape[1]
        self.pop = (R > 0).sum(0)
        self.K_svd = pure_svd_kernel(R, f)
        self.K_knn = itemknn_kernel(R, k)
        idf = np.log(self.n_items / (1.0 + G.sum(0)))
        self.X = G * idf
        self.genre_names = None  # optional, set by caller for readable genre reasons

    # ---------------------------------------------------------------- scoring
    def _vectors(self, liked):
        x = np.zeros(self.n_items)
        for i, r in liked.items():
            x[i] = r
        return x

    def _content(self, x):
        row = content_scores(x[None, :], self.G)
        return row[0]

    def scores(self, liked, model):
        x = self._vectors(liked)
        if model == "popularity":
            return self.pop.astype(float)
        if model == "puresvd":
            return x @ self.K_svd
        if model == "item_knn":
            return (x > 0).astype(float) @ self.K_knn
        if model == "content":
            return self._content(x)
        if model == "hybrid":
            s = zrow((x @ self.K_svd)[None, :])[0]
            c = zrow(self._content(x)[None, :])[0]
            return s + self.w * c
        raise ValueError(f"unknown model: {model}")

    def contributions(self, liked, model, i):
        """{liked item j: contribution of j to the score of i} for the linear models."""
        if model in ("puresvd", "hybrid"):
            return {j: float(r * self.K_svd[j, i]) for j, r in liked.items()}
        if model == "item_knn":
            return {j: float(self.K_knn[j, i]) for j in liked}
        return {}

    def _shared_genres(self, liked, i):
        liked_hi = [j for j, r in liked.items() if r >= REL] or list(liked)
        profile = self.G[liked_hi].sum(0)
        shared = np.flatnonzero((profile > 0) & (self.G[i] > 0))
        order = shared[np.argsort(-profile[shared])]
        return [self.genre_names[g] for g in order[:3]] if self.genre_names else []

    # ---------------------------------------------------------------- public API
    def recommend(self, liked, model="hybrid", n=10):
        """liked: {0-based item id: rating 1..5}. Returns a list of dicts, best first."""
        if not liked or model == "popularity":
            model_used = "popularity"
            scores = self.pop.astype(float)
        else:
            model_used = model
            scores = self.scores(liked, model)
        scores = scores.copy()
        for j in liked:
            scores[j] = -np.inf  # never recommend what the user already gave
        scores = scores - np.arange(self.n_items) * 1e-9
        top = np.argsort(-scores)[:n]
        out = []
        for i in top:
            i = int(i)
            reasons = []
            if model_used in ("puresvd", "item_knn", "hybrid"):
                contrib = self.contributions(liked, model_used, i)
                pos = {j: c for j, c in contrib.items() if c > 0}
                total = sum(pos.values())
                for j, c in sorted(pos.items(), key=lambda kv: -kv[1])[:2]:
                    reasons.append({"movie_id": j + 1, "title": self.items.at[j, "title"],
                                    "share": round(c / total, 3) if total > 0 else 0.0})
            genres = self._shared_genres(liked, i) if model_used in ("content", "hybrid") and liked else []
            out.append({
                "movie_id": i + 1,
                "title": self.items.at[i, "title"],
                "year": self.items.at[i, "year"],
                "genres": self.items.at[i, "genres"],
                "n_ratings": int(self.pop[i]),
                "score": float(scores[i] + i * 1e-9),
                "reasons": reasons,
                "shared_genres": genres,
                "reason_text": self._text(model_used, reasons, genres, int(self.pop[i])),
            })
        return out

    @staticmethod
    def _text(model, reasons, genres, n_ratings):
        if model == "popularity":
            return f"Rated by {n_ratings} users, one of the most-rated movies."
        parts = []
        if reasons:
            names = " and ".join(r["title"] for r in reasons)
            parts.append(f"Because you liked {names}")
        if genres:
            parts.append("matches your taste in " + ", ".join(genres))
        if not parts:
            return "Similar to the movies you picked."
        return "; ".join(parts) + "."
