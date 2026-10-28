"""
Makes the figures of the report:  python make_figures.py  (about 6 minutes)

results/fig_missing.pdf    fraction of missing values per feature (raw data)
results/fig_threshold.pdf  F1 as a function of the decision threshold (one 80/20 split)
results/fig_lambda.pdf     train and validation F1 as a function of lambda (5-fold CV)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import load_data, clean, fit, transform
from cv import cross_validate, f1_score, make_logreg

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a85"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e3",
        "grid.linewidth": 0.5,
    }
)
SIZE = (3.4, 2.1)  # one column of the IEEE two-column template, in inches


def missing_figure(x):
    """Missing fraction of every raw feature, sorted."""
    nan_frac = np.sort(np.isnan(x).mean(axis=0))
    plt.figure(figsize=SIZE)
    plt.plot(np.arange(1, len(nan_frac) + 1), nan_frac, color=BLUE, lw=1.6)
    plt.axhline(0.5, color=GRAY, ls="--", lw=0.8)
    plt.text(3, 0.54, "threshold 50%", fontsize=7)
    plt.xlabel("feature (sorted by missing fraction)")
    plt.ylabel("fraction missing")
    plt.tight_layout()
    plt.savefig("results/fig_missing.pdf")
    print(f"features > 50% missing: {np.sum(nan_frac > 0.5)} of {len(nan_frac)}")


def threshold_figure(Xc, names, y):
    """F1 on a held-out 20% for every threshold on the predicted probability."""
    perm = np.random.default_rng(0).permutation(len(y))
    va, tr = perm[: len(y) // 5], perm[len(y) // 5 :]
    p = fit(Xc[tr], names)
    A, B = transform(Xc[tr], names, p), transform(Xc[va], names, p)
    w = make_logreg()(y[tr], A)
    prob = 1 / (1 + np.exp(-(B @ w)))

    ts = np.linspace(0.02, 0.8, 79)
    f1s = np.array([f1_score(y[va], np.where(prob > t, 1, -1)) for t in ts])
    t_best, f1_best = ts[np.argmax(f1s)], f1s.max()
    f1_half = f1_score(y[va], np.where(prob > 0.5, 1, -1))

    plt.figure(figsize=SIZE)
    plt.plot(ts, f1s, color=BLUE, lw=1.6)
    plt.axvline(0.5, color=GRAY, ls="--", lw=0.8)
    plt.axvline(t_best, color=GRAY, ls=":", lw=0.8)
    plt.text(0.51, f1_half + 0.03, f"default 0.5\nF1 = {f1_half:.2f}", fontsize=7)
    plt.text(
        t_best + 0.02,
        f1_best + 0.01,
        f"best {t_best:.2f}, F1 = {f1_best:.2f}",
        fontsize=7,
    )
    plt.xlabel("decision threshold on predicted probability")
    plt.ylabel("validation F1")
    plt.ylim(0, f1_best + 0.08)
    plt.tight_layout()
    plt.savefig("results/fig_threshold.pdf")
    print(
        f"threshold 0.5: F1 {f1_half:.3f}   best threshold {t_best:.2f}: F1 {f1_best:.3f}"
    )


def lambda_figure(Xc, names, y):
    """Train vs validation F1 (5-fold CV) for regularized logistic regression."""
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    res = np.array([cross_validate(make_logreg(lam), Xc, names, y) for lam in lambdas])

    plt.figure(figsize=SIZE)
    plt.semilogx(lambdas, res[:, 2], "o-", color=ORANGE, lw=1.6, ms=3.5, label="train")
    plt.errorbar(
        lambdas,
        res[:, 0],
        yerr=res[:, 1],
        fmt="o-",
        color=BLUE,
        lw=1.6,
        ms=3.5,
        capsize=2,
        label="validation (± std over 5 folds)",
    )
    plt.xlabel(r"regularization $\lambda$")
    plt.ylabel("F1")
    plt.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    plt.savefig("results/fig_lambda.pdf")
    for lam, (m, s, t) in zip(lambdas, res):
        print(f"lambda {lam:g}: train {t:.3f}  val {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    x_train, x_test, y_train, train_ids, test_ids, names = load_data("dataset/")
    missing_figure(x_train)
    Xc, kept = clean(x_train, names)
    threshold_figure(Xc, kept, y_train)
    lambda_figure(Xc, kept, y_train)
