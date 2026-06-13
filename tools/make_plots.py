#!/usr/bin/env python3
"""Render ml_lib's result plots from the CSVs written by tools/export_viz.c.

Run from the repo root after building/running export_viz:
    python tools/make_plots.py
Outputs PNGs into docs/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs"
os.makedirs(OUT, exist_ok=True)

# Shared look: clean, slightly "designed".
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
ACCENT = "#4f46e5"   # indigo
WARM   = "#ef4444"   # red


def decision_boundary():
    g = np.genfromtxt("viz_grid.csv", delimiter=",", names=True)
    n = int(round(len(g["x1"]) ** 0.5))
    X1 = g["x1"].reshape(n, n)
    X2 = g["x2"].reshape(n, n)
    P  = g["prob"].reshape(n, n)

    d = np.genfromtxt("data/xor2d.csv", delimiter=",", names=True)
    lab = d["label"].astype(int)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    cf = ax.contourf(X1, X2, P, levels=np.linspace(0, 1, 21),
                     cmap="RdBu_r", alpha=0.85)
    ax.contour(X1, X2, P, levels=[0.5], colors="k", linewidths=1.4)
    ax.scatter(d["x1"][lab == 0], d["x2"][lab == 0], s=18, c="#1d4ed8",
               edgecolor="white", linewidth=0.4, label="class 0")
    ax.scatter(d["x1"][lab == 1], d["x2"][lab == 1], s=18, c="#b91c1c",
               edgecolor="white", linewidth=0.4, label="class 1")
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(class 1)")
    ax.set_title("Learned decision boundary (2-8-1 MLP)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/decision_boundary.png", bbox_inches="tight")
    plt.close(fig)


def loss_curves():
    s = np.genfromtxt("viz_loss_sgd.csv",  delimiter=",", names=True)
    a = np.genfromtxt("viz_loss_adam.csv", delimiter=",", names=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(s["epoch"], s["loss"], color=WARM,   lw=2, label="SGD (lr=0.5)")
    ax.plot(a["epoch"], a["loss"], color=ACCENT, lw=2, label="Adam (lr=0.01)")
    ax.set_yscale("log")
    ax.set_title("Training loss: SGD vs Adam (XOR-of-signs)")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean loss (log scale)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/loss_curves.png", bbox_inches="tight")
    plt.close(fig)


def iris_confusion():
    c = np.genfromtxt("viz_iris_confusion.csv", delimiter=",", names=True)
    M = np.zeros((3, 3), dtype=int)
    for t, p, n in zip(c["true"].astype(int), c["pred"].astype(int),
                       c["count"].astype(int)):
        M[t, p] = n
    classes = ["setosa", "versicolor", "virginica"]
    acc = 100.0 * np.trace(M) / M.sum()

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(M, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "#1e293b",
                    fontsize=14, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Iris validation confusion ({acc:.1f}% held-out)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{OUT}/iris_confusion.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    decision_boundary()
    loss_curves()
    iris_confusion()
    print("wrote docs/decision_boundary.png, docs/loss_curves.png, "
          "docs/iris_confusion.png")
