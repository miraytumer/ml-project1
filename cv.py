"""
Evaluation: F1 score, decision threshold, 5-fold cross-validation, and one training
function per required method (each returns train(y, tX) -> w).

    Xc, cols = clean(x_train, names)
    cross_validate(make_logreg(), Xc, cols, y_train)              # 0.424
    cross_validate(make_logreg(), Xc, cols, y_train, pairs=True)  # final model, 0.429
"""

import numpy as np
from preprocessing import fit, transform, fit_pairs, add_pairs
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)


def f1_score(y_true, y_pred):
    """F1 for the positive class; labels in {-1, 1}."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == -1))
    fn = np.sum((y_pred == -1) & (y_true == 1))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def best_threshold(y, scores):
    """Threshold on tX @ w maximizing F1; candidates flag the top 1% ... 50% of people."""
    grid = np.quantile(scores, np.linspace(0.50, 0.99, 50))
    f1s = [f1_score(y, np.where(scores > t, 1, -1)) for t in grid]
    return grid[int(np.argmax(f1s))]


def predict(tX, w, threshold):
    """Labels in {-1, 1}."""
    return np.where(tX @ w > threshold, 1, -1)


# ---------- training functions (MSE methods on -1/+1, logistic methods on 0/1) ----------


def make_mse_gd(max_iters=500, gamma=0.05):
    return lambda y, tX: mean_squared_error_gd(
        y, tX, np.zeros(tX.shape[1]), max_iters, gamma
    )[0]


def make_mse_sgd(max_iters=20000, gamma=0.001, seed=0):
    def train(y, tX):
        np.random.seed(seed)
        return mean_squared_error_sgd(y, tX, np.zeros(tX.shape[1]), max_iters, gamma)[0]

    return train


def make_least_squares():
    return lambda y, tX: least_squares(y, tX)[0]


def make_ridge(lambda_=1e-4):
    return lambda y, tX: ridge_regression(y, tX, lambda_)[0]


def make_logistic(max_iters=500, gamma=0.5):
    return lambda y, tX: logistic_regression(
        (y == 1).astype(float), tX, np.zeros(tX.shape[1]), max_iters, gamma
    )[0]


def make_logreg(lambda_=1e-4, max_iters=500, gamma=0.5):
    return lambda y, tX: reg_logistic_regression(
        (y == 1).astype(float), tX, lambda_, np.zeros(tX.shape[1]), max_iters, gamma
    )[0]


# ---------- cross-validation ----------


def cross_validate(train_fn, Xc, names, y, k=5, seed=0, pairs=False, K=15):
    """k-fold CV. Preprocessing, interactions and threshold are fitted on the training folds
    only. pairs=True adds the pairwise products of the K strongest columns (final model).
    Returns (mean val F1, std val F1, mean train F1)."""
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), k)
    val, train = [], []
    for i in range(k):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        p = fit(Xc[tr], names)
        tX_tr, tX_va = transform(Xc[tr], names, p), transform(Xc[va], names, p)
        if pairs:  # same steps as run.py
            pp = fit_pairs(tX_tr, train_fn(y[tr], tX_tr), K)
            tX_tr, tX_va = add_pairs(tX_tr, pp), add_pairs(tX_va, pp)
        w = train_fn(y[tr], tX_tr)
        t = best_threshold(y[tr], tX_tr @ w)
        val.append(f1_score(y[va], predict(tX_va, w, t)))
        train.append(f1_score(y[tr], predict(tX_tr, w, t)))
    m, s, t = float(np.mean(val)), float(np.std(val)), float(np.mean(train))
    print(f"train F1 {t:.3f}   val F1 {m:.3f} ± {s:.3f}")
    return m, s, t
