"""Loading and shaping the MovieLens 100K data.

User and item ids are converted to 0-based indices everywhere inside the package.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

N_USERS, N_ITEMS = 943, 1682


def load_ratings(data_dir: str, name: str = "u.data") -> pd.DataFrame:
    """Read a tab-separated ratings file (u.data, u1.base, u1.test, ...) with 0-based ids."""
    df = pd.read_csv(os.path.join(data_dir, name), sep="\t", header=None, names=["u", "i", "r", "t"])
    df["u"] -= 1
    df["i"] -= 1
    return df


def to_dense(df: pd.DataFrame) -> np.ndarray:
    """Dense user x item rating matrix; 0 means 'not rated'."""
    R = np.zeros((N_USERS, N_ITEMS))
    R[df["u"].values, df["i"].values] = df["r"].values
    return R


def load_genre_names(data_dir: str) -> list[str]:
    names = []
    with open(os.path.join(data_dir, "u.genre"), encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line.split("|")[0])
    return names


def load_items(data_dir: str):
    """Return (items DataFrame, genre matrix G of shape (N_ITEMS, 19), genre names)."""
    raw = pd.read_csv(os.path.join(data_dir, "u.item"), sep="|", header=None, encoding="latin-1")
    G = raw.iloc[:, 5:24].values.astype(float)
    names = load_genre_names(data_dir)
    items = pd.DataFrame({"item": raw[0].values - 1, "title": raw[1].values})
    items["year"] = items["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    items["genres"] = [[names[g] for g in np.flatnonzero(row)] for row in G]
    return items, G, names
