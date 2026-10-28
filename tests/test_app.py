"""API tests. They need the MovieLens 100K files (data/ml-100k or ML100K_DIR) and are skipped otherwise."""
import os
from pathlib import Path

import numpy as np
import pytest

DATA = Path(os.environ.get("ML100K_DIR", Path(__file__).resolve().parents[1] / "data" / "ml-100k"))
pytestmark = pytest.mark.skipif(not (DATA / "u.data").exists(), reason="MovieLens 100K data not found")


@pytest.fixture(scope="module")
def client():
    os.environ["ML100K_DIR"] = str(DATA)
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "movies": 1682}


def test_search_finds_toy_story_first(client):
    hits = client.get("/api/movies/search", params={"q": "toy story"}).json()
    assert hits[0]["movie_id"] == 1 and "Toy Story" in hits[0]["title"]


def test_popular_is_sorted_by_ratings(client):
    pop = client.get("/api/movies/popular", params={"limit": 10}).json()
    counts = [m["n_ratings"] for m in pop]
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize("model", ["hybrid", "puresvd", "item_knn", "content", "popularity"])
def test_recommend_excludes_liked_and_explains(client, model):
    liked = [{"movie_id": 1, "rating": 5}, {"movie_id": 50, "rating": 5}, {"movie_id": 181, "rating": 4}]
    r = client.post("/api/recommend", json={"liked": liked, "model": model, "n": 10})
    assert r.status_code == 200
    recs = r.json()["recommendations"]
    assert len(recs) == 10
    assert not ({m["movie_id"] for m in recs} & {1, 50, 181})
    assert all(m["reason_text"] for m in recs)
    for m in recs:
        assert sum(x["share"] for x in m["reasons"]) <= 1.0001


def test_empty_list_falls_back_to_popularity(client):
    r = client.post("/api/recommend", json={"liked": [], "model": "hybrid"}).json()
    assert r["note"] and len(r["recommendations"]) == 10


def test_bad_requests_are_rejected(client):
    dup = {"liked": [{"movie_id": 1, "rating": 5}, {"movie_id": 1, "rating": 4}]}
    assert client.post("/api/recommend", json=dup).status_code == 422
    assert client.post("/api/recommend", json={"liked": [], "model": "nope"}).status_code == 422
    assert client.post("/api/recommend", json={"liked": [{"movie_id": 9999, "rating": 5}]}).status_code == 422


def test_fold_in_matches_training_user(client):
    """Feeding a training user's own ratings through the app path reproduces PureSVD's row for that user."""
    from app.main import STATE
    from recsys.models import pure_svd
    rec = STATE["rec"]
    u = 12
    liked = {int(i): float(rec.R[u, i]) for i in np.flatnonzero(rec.R[u])}
    expected = pure_svd(rec.R, 20)[u]
    assert np.allclose(rec.scores(liked, "puresvd"), expected, atol=1e-8)
