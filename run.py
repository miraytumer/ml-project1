"""
Model: regularized logistic regression (lambda=1e-4, 500 GD steps, gamma=0.5) on all cleaned
features (see preprocessing.py) plus the pairwise products of the 15 columns with the largest
weights in a first model (120 extra columns). Decision threshold chosen to maximize F1 on the
training data.
5-fold cross-validated F1: 0.429 +- 0.003. AIcrowd F1: 0.442.
"""

import numpy as np
from helpers import create_csv_submission
from preprocessing import load_data, prepare, fit_pairs, add_pairs
from cv import make_logreg, best_threshold, predict

# final choices (selected by cross-validation, see README / report)
LAMBDA = 1e-4
MAX_ITERS = 500
GAMMA = 0.5
K_PAIRS = 15  # number of strongest columns whose pairwise products are added


def main():
    # 1. data
    x_train, x_test, y_train, train_ids, test_ids, names = load_data("data/")

    # 2. preprocessing: all features except admin / leakage / redundant; statistics learned
    #    on the full training set, applied to both sets
    tX_train, tX_test, p = prepare(x_train, x_test, names)

    # 3. first model, used only to find the strongest columns for the interactions
    train = make_logreg(LAMBDA, MAX_ITERS, GAMMA)
    w0 = train(y_train, tX_train)
    pp = fit_pairs(tX_train, w0, K_PAIRS)
    A, B = add_pairs(tX_train, pp), add_pairs(tX_test, pp)
    print(
        f"preprocessed: {tX_train.shape[1]} columns + {len(pp['pairs'])} interactions"
    )

    # 4. final model on all training data, threshold that maximizes training F1
    w = train(y_train, A)
    t = best_threshold(y_train, A @ w)

    # 5. predict and write the submission
    y_pred = predict(B, w, t)
    create_csv_submission(test_ids, y_pred, "submission.csv")
    print(
        f"submission.csv written: {len(y_pred)} predictions, {np.mean(y_pred == 1):.1%} positive"
    )


if __name__ == "__main__":
    main()
