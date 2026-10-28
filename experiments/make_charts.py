#!/usr/bin/env python3
"""Figures for the report.

Usage: python experiments/make_charts.py results results/figures
"""
import sys, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res_dir, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
R = json.load(open(os.path.join(res_dir, "results.json")))
S, ST = R["summary"], R["stats"]
cnt = np.load(os.path.join(res_dir, "item_counts.npy"))

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 220,
})


def clean(ax, grid_axis):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---------- Fig 1: dataset
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={"wspace": 0.32})
vals = [ST["rating_dist"][str(k)] for k in range(1, 6)]
a.bar(range(1, 6), vals, width=0.62, color=BLUE)
for x, v in zip(range(1, 6), vals):
    a.text(x, v + 600, f"{v:,}", ha="center", va="bottom", fontsize=7.5, color=INK2)
a.set_xticks(range(1, 6)); a.set_xlabel("Star rating"); a.set_ylabel("Number of ratings")
a.set_ylim(0, 40000); a.set_yticks([0, 10000, 20000, 30000, 40000]); a.set_yticklabels(["0", "10k", "20k", "30k", "40k"])
a.set_title("Rating distribution", loc="left", fontsize=9, color=INK, fontweight="bold")
clean(a, "y")
srt = np.sort(cnt)[::-1]
cum = np.concatenate([[0], np.cumsum(srt) / srt.sum()])
xs = np.arange(len(cum)) / len(srt)
b.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 3)))
b.plot(xs, cum, color=BLUE, lw=2)
i20 = int(0.2 * len(srt))
b.plot([xs[i20]], [cum[i20]], "o", ms=7, color=BLUE, mec="white", mew=1.5)
b.annotate(f"Top 20% of\nmovies hold\n{cum[i20]*100:.0f}% of ratings", (xs[i20], cum[i20]), xytext=(0.40, 0.64),
           fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
b.text(0.60, 0.30, "equal-share line", fontsize=7.5, color=MUTED)
b.set_xlim(0, 1); b.set_ylim(0, 1.02)
b.set_xticks([0, .2, .4, .6, .8, 1]); b.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
b.set_yticks([0, .25, .5, .75, 1]); b.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
b.set_xlabel("Movies, most-rated first"); b.set_ylabel("Cumulative share of ratings")
b.set_title("Long-tail popularity", loc="left", fontsize=9, color=INK, fontweight="bold")
clean(b, "y")
fig.savefig(os.path.join(out, "fig1_dataset.png"), bbox_inches="tight"); plt.close(fig)

# ---------- Fig 2: RMSE dot plot
order = ["Item-kNN", "User-kNN", "Matrix factorisation (ALS)", "Bias baseline", "Global mean"]
labels = {"Matrix factorisation (ALS)": "Matrix factorisation (ALS)"}
fig, ax = plt.subplots(figsize=(6.6, 2.5))
for y, m in enumerate(order[::-1]):
    mu, sd = S[m]["RMSE"]["mean"], S[m]["RMSE"]["std"]
    ax.plot([mu - sd, mu + sd], [y, y], color=BLUE, lw=1.6, solid_capstyle="round")
    ax.plot([mu], [y], "o", ms=8, color=BLUE, mec="white", mew=1.5)
    ax.text(mu + sd + 0.006, y, f"{mu:.4f}", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(order))); ax.set_yticklabels([labels.get(m, m) for m in order[::-1]], color=INK2)
ax.set_xlim(0.88, 1.20); ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xlabel("RMSE on held-out ratings (lower is better); dot = mean, line = ±1 SD over 5 folds. Axis starts at 0.88.")
clean(ax, "x")
fig.savefig(os.path.join(out, "fig2_rmse.png"), bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: NDCG@10 bars
rank_models = ["Hybrid (PureSVD + genres)", "PureSVD", "Item-kNN (implicit)", "Popularity", "Item-kNN", "User-kNN",
               "Matrix factorisation (ALS)", "Bias baseline", "Content-based (genres)"]
is_rank = {"Hybrid (PureSVD + genres)", "PureSVD", "Item-kNN (implicit)", "Popularity", "Content-based (genres)"}
fig, ax = plt.subplots(figsize=(6.6, 3.5))
for y, m in enumerate(rank_models[::-1]):
    mu, sd = S[m]["NDCG@10"]["mean"], S[m]["NDCG@10"]["std"]
    c = ORANGE if m in is_rank else BLUE
    ax.barh(y, mu, height=0.55, color=c)
    ax.plot([mu - sd, mu + sd], [y, y], color=INK2, lw=0.9)
    ax.text(mu + sd + 0.008, y, f"{mu:.3f}", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(rank_models))); ax.set_yticklabels(rank_models[::-1], color=INK2)
ax.set_xlim(0, 0.50); ax.set_xlabel("NDCG@10, full-catalogue ranking (higher is better); thin line = ±1 SD over 5 folds")
clean(ax, "x")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="Rating-prediction models"),
                   Patch(color=ORANGE, label="Popularity, implicit-feedback, genre and hybrid models")],
          loc="lower right", frameon=False, fontsize=7.5, labelcolor=INK2)
fig.savefig(os.path.join(out, "fig3_ndcg.png"), bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4: coverage vs NDCG
fig, ax = plt.subplots(figsize=(6.4, 3.3))
offs = {"Hybrid (PureSVD + genres)": None, "Content-based (genres)": (0, 9, "center"), "PureSVD": (8, 4, "left"), "Item-kNN (implicit)": (-9, -12, "right"), "Popularity": (8, 5, "left"),
        "Item-kNN": (8, -3, "left"), "User-kNN": (-4, 9, "right"), "Matrix factorisation (ALS)": (-8, -14, "right"),
        "Bias baseline": (8, -12, "left")}
for m in rank_models:
    x, y = S[m]["Coverage"]["mean"] * 100, S[m]["NDCG@10"]["mean"]
    c = ORANGE if m in is_rank else BLUE
    ax.plot([x], [y], "o", ms=9, color=c, mec="white", mew=1.5)
    if offs[m] is None:
        continue
    dx, dy, ha = offs[m]
    ax.annotate("PureSVD ≈ Hybrid" if m == "PureSVD" else m, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=7.8, color=INK2)
ax.set_xlim(0, 34); ax.set_ylim(0.04, 0.44)
ax.set_xlabel("Catalogue coverage@10 (% of the 1,682 movies that appear in anyone's top-10)")
ax.set_ylabel("NDCG@10")
clean(ax, "both")
fig.savefig(os.path.join(out, "fig4_coverage.png"), bbox_inches="tight"); plt.close(fig)
print("figures written to", out)

# ---------- Fig 5: cold start (needs cold_start.json from run_cold_start.py)
cs_path = os.path.join(res_dir, "cold_start.json")
if os.path.exists(cs_path):
    CS = json.load(open(cs_path))
    cs = CS["summary"]
    AQUA, VIOLET = "#1baf7a", "#6250d6"
    series = [  # name, colour, marker, linestyle
        ("Popularity", INK2, "s", "-"),
        ("Content-based (genres)", AQUA, "^", "-"),
        ("Item-kNN (implicit)", ORANGE, "D", "-"),
        ("PureSVD", BLUE, "o", "-"),
        ("Hybrid (PureSVD + genres)", VIOLET, "o", (0, (2, 2))),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for name, col, mk, ls in series:
        ns = sorted(int(n) for n in cs[name])
        ys = [cs[name][str(n)]["NDCG@10"]["mean"] for n in ns]
        ax.plot(ns, ys, color=col, lw=2, ls=ls, marker=mk, ms=7, mec="white", mew=1.2, zorder=3)
    end = lambda name: cs[name]["10"]["NDCG@10"]["mean"]
    labels = [("PureSVD ≈ Hybrid", 0.5 * (end("PureSVD") + end("Hybrid (PureSVD + genres)")), BLUE),
              ("Item-kNN (implicit)", end("Item-kNN (implicit)"), ORANGE),
              ("Popularity", end("Popularity"), INK2),
              ("Content-based (genres)", end("Content-based (genres)"), AQUA)]
    for text, y, col in labels:
        ax.text(10.35, y, text, va="center", fontsize=8, color=INK2)
    y3 = cs["PureSVD"]["3"]["NDCG@10"]["mean"]
    ax.annotate("PureSVD passes popularity\nat about 3 ratings", (3, y3), xytext=(4.3, 0.065), fontsize=7.8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(-0.3, 15.2); ax.set_ylim(0, 0.25)
    ax.set_xticks([0, 1, 3, 5, 10])
    ax.set_xlabel("Ratings given by the new user (n)  |  with n = 0 only popularity can be used")
    ax.set_ylabel("NDCG@10")
    clean(ax, "y")
    fig.savefig(os.path.join(out, "fig5_cold_start.png"), bbox_inches="tight"); plt.close(fig)
    print("cold-start figure written")
