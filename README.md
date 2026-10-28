# Movie recommender: benchmark, cold-start study and web app

Recommender systems project on the MovieLens 100K dataset (943 users, 1,682 movies, 100,000 ratings).
It contains three things:

1. **A benchmark** of ten recommenders (baselines, user/item kNN, matrix factorisation, PureSVD, genre content-based, a hybrid)
   with five-fold cross-validation, measuring rating error (RMSE, MAE) and top-10 ranking quality (P@10, R@10, NDCG@10, coverage).
2. **A cold-start study** that shows how many ratings a new user must give before personalised models beat a plain popularity list.
3. **A web app** (FastAPI + one HTML page) where you pick a few movies you like and get ten recommendations,
   each with a "because you liked ..." explanation.

## Results at a glance

Rating prediction (lower is better):

| Model | RMSE (mean ± SD, 5 folds) | MAE |
|---|---|---|
| Global mean | 1.1256 ± 0.0174 | 0.9447 |
| Bias baseline | 0.9457 ± 0.0087 | 0.7499 |
| User-kNN | 0.9221 ± 0.0078 | 0.7263 |
| Item-kNN | 0.9128 ± 0.0083 | 0.7181 |
| Matrix factorisation (ALS) | 0.9243 ± 0.0065 | 0.7267 |

Top-10 ranking, full catalogue, relevant = rated 4 or 5 in the test set (higher is better):

| Model | NDCG@10 | P@10 | R@10 | Coverage@10 |
|---|---|---|---|---|
| Hybrid (PureSVD + genres) | 0.397 | 0.299 | 0.274 | 20.0% |
| PureSVD | 0.392 | 0.295 | 0.271 | 19.7% |
| Item-kNN (implicit) | 0.336 | 0.261 | 0.238 | 20.3% |
| Popularity | 0.198 | 0.157 | 0.136 | 3.5% |
| Item-kNN | 0.121 | 0.104 | 0.066 | 9.5% |
| User-kNN | 0.110 | 0.100 | 0.061 | 6.2% |
| Matrix factorisation (ALS) | 0.100 | 0.089 | 0.054 | 21.9% |
| Bias baseline | 0.097 | 0.094 | 0.057 | 2.5% |
| Content-based (genres) | 0.077 | 0.060 | 0.042 | 29.4% |

Cold start: NDCG@10 for new users who have given *n* ratings (mean of 15 runs; with n=0 only popularity can be used):

| Model | n=0 | n=1 | n=3 | n=5 | n=10 |
|---|---|---|---|---|---|
| Popularity | 0.134 | 0.134 | 0.134 | 0.134 | 0.133 |
| Content-based (genres) | - | 0.026 | 0.023 | 0.025 | 0.030 |
| Item-kNN (implicit) | - | 0.118 | 0.135 | 0.156 | 0.181 |
| PureSVD | - | 0.128 | 0.164 | 0.185 | 0.220 |
| Hybrid (PureSVD + genres) | - | 0.127 | 0.164 | 0.187 | 0.222 |

Short version: item-kNN predicts ratings best, but for ranking a top-10 list, models trained on ratings do worse than a popularity
list, while PureSVD and implicit item-kNN do far better. A new user needs about three ratings before PureSVD beats the popularity list;
genre information alone never does. Details and discussion are in the report.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 1. Get the data (not included, see "Data and licence"): unzip so that data/ml-100k/u.data exists
#    https://grouplens.org/datasets/movielens/100k/

# 2. Run the web app, then open http://127.0.0.1:8000
uvicorn app.main:app --reload

# 3. Reproduce the experiments (a few minutes each)
python experiments/run_benchmark.py data/ml-100k results
python experiments/run_cold_start.py data/ml-100k results
python experiments/make_charts.py results results/figures

# 4. Tests (API tests are skipped if the data is missing)
pytest
```

If the data lives elsewhere, set `ML100K_DIR=/path/to/ml-100k` before starting the app or running the tests.

## Project layout

```
app/                  FastAPI app (main.py) and the web page (static/index.html)
src/recsys/           data.py (loading), models.py (algorithms), metrics.py (evaluation), recommend.py (fold-in + explanations)
experiments/          run_benchmark.py, run_cold_start.py, make_charts.py
results/              results.json, summary.csv, cold_start.json, cold_start_ndcg.csv, figures/
tests/                unit tests (synthetic data) and API tests (need the dataset)
```

## How the explanations work

PureSVD and implicit item-kNN are linear item-item models. For a user with rating vector *x* the score of movie *i* is

    score(i) = sum over the user's movies j of  x_j * K[j, i]

where K = V_f V_fᵀ for PureSVD (V_f = top-f right singular vectors of the rating matrix) and K = the neighbour-similarity matrix for
item-kNN. A brand-new user is "folded in" by multiplying their ratings with K, with no retraining. Because the score is a sum over the
user's movies, the share of movie *j* in a recommendation is `x_j K[j,i] / sum of the positive terms`; the app shows the top two.
A unit test checks that folding in an existing user's own ratings reproduces PureSVD's row for that user exactly.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness check |
| GET | `/api/models` | available models and their descriptions |
| GET | `/api/movies/search?q=toy&limit=10` | title search, most-rated first |
| GET | `/api/movies/popular?limit=20` | most-rated movies |
| POST | `/api/recommend` | body: `{"liked": [{"movie_id": 1, "rating": 5}], "model": "hybrid", "n": 10}` |

`model` is one of `hybrid`, `puresvd`, `item_knn`, `content`, `popularity`. `movie_id` is the 1-based MovieLens id.
Interactive docs are served at `/docs`.

## Evaluation protocol (short)

* Official five folds `u1..u5` (80,000 train / 20,000 test). Hyper-parameters are tuned per fold on a 10% hold-out of that fold's
  training data; test data is never used for tuning.
* Ranking: every movie the user has not rated in training is a candidate (about 1,600 per user); a test movie is relevant if its rating is 4 or 5.
* Cold start: for each simulated new user the last 10 of a random ordering of their ratings are held out; the first *n* are revealed.
  Their other ratings are removed from the candidate list. Three random repeats of five user groups.

## Limitations

One small, old, movies-only dataset; random (not temporal) splits; offline metrics only, so no claim about real users;
coarse hyper-parameter grids; genre is the only item feature used by the content-based model.

## Data and licence

The MovieLens 100K data is **not** included: its licence does not allow redistribution. Download it from GroupLens and cite
F. M. Harper and J. A. Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), Article 19, 2015.
