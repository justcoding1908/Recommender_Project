"""FastAPI app: recommend movies for a new user from a few liked movies, with explanations.

Run:  uvicorn app.main:app --reload        (from the project root)
Data: set ML100K_DIR or put the unzipped dataset in data/ml-100k
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recsys.data import load_items, load_ratings, to_dense  # noqa: E402
from recsys.recommend import MODELS, Recommender  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
STATE: dict = {}


def build_state(data_dir: str) -> dict:
    items, G, genre_names = load_items(data_dir)
    R = to_dense(load_ratings(data_dir))
    rec = Recommender(R, G, items)
    rec.genre_names = genre_names
    return {"rec": rec, "items": items, "pop": rec.pop}


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = os.environ.get("ML100K_DIR", str(ROOT / "data" / "ml-100k"))
    if not Path(data_dir, "u.data").exists():
        raise RuntimeError(
            f"MovieLens 100K not found at {data_dir}. Download ml-100k.zip from "
            "https://grouplens.org/datasets/movielens/100k/ and unzip it into data/, "
            "or set the ML100K_DIR environment variable.")
    STATE.update(build_state(data_dir))
    yield
    STATE.clear()


app = FastAPI(title="MovieLens recommender with explanations", version="1.0", lifespan=lifespan)


class LikedMovie(BaseModel):
    movie_id: int = Field(..., ge=1, le=1682, description="MovieLens movie id (1-based)")
    rating: int = Field(5, ge=1, le=5)


class RecommendRequest(BaseModel):
    liked: list[LikedMovie] = Field(default_factory=list, max_length=50)
    model: Literal["hybrid", "puresvd", "item_knn", "content", "popularity"] = "hybrid"
    n: int = Field(10, ge=1, le=30)


def movie_card(i: int) -> dict:
    items, pop = STATE["items"], STATE["pop"]
    return {"movie_id": i + 1, "title": items.at[i, "title"], "year": items.at[i, "year"],
            "genres": items.at[i, "genres"], "n_ratings": int(pop[i])}


@app.get("/api/health")
def health():
    return {"status": "ok", "movies": int(len(STATE["items"]))}


@app.get("/api/models")
def models():
    return [{"id": k, "description": v} for k, v in MODELS.items()]


@app.get("/api/movies/search")
def search(q: str = Query(..., min_length=2, max_length=60), limit: int = Query(10, ge=1, le=30)):
    items, pop = STATE["items"], STATE["pop"]
    mask = items["title"].str.contains(q, case=False, regex=False)
    idx = np.flatnonzero(mask.values)
    idx = idx[np.argsort(-pop[idx])][:limit]  # most-rated matches first
    return [movie_card(int(i)) for i in idx]


@app.get("/api/movies/popular")
def popular(limit: int = Query(20, ge=1, le=50)):
    return [movie_card(int(i)) for i in np.argsort(-STATE["pop"])[:limit]]


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    liked: dict[int, float] = {}
    for m in req.liked:
        if m.movie_id - 1 in liked:
            raise HTTPException(422, f"movie {m.movie_id} listed twice")
        liked[m.movie_id - 1] = float(m.rating)
    recs = STATE["rec"].recommend(liked, model=req.model, n=req.n)
    note = None
    if not liked and req.model != "popularity":
        note = "No movies picked yet, so the most popular movies are shown."
    return {"model": req.model, "note": note, "recommendations": recs}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
