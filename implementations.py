"""Basic machine learning methods for EPFL ML Project 1."""

import numpy as np


def _mse(y, tx, w):
    """Mean squared error loss with the 1/2 factor: (1/2N) * sum (y - tx w)^2."""
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Returns:
        (w, loss): last weight vector and its MSE loss.
    """
    w = initial_w.copy()
    N = len(y)
    for _ in range(max_iters):
        grad = -tx.T @ (y - tx @ w) / N
        w = w - gamma * grad
    return w, _mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).

    Returns:
        (w, loss): last weight vector and its MSE loss.
    """
    w = initial_w.copy()
    N = len(y)
    for _ in range(max_iters):
        i = np.random.randint(N)  # sample one datapoint
        grad = -tx[i] * (y[i] - tx[i] @ w)
        w = w - gamma * grad
    return w, _mse(y, tx, w)


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Returns:
        (w, loss): optimal weights and their MSE loss.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, _mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Returns:
        (w, loss): weights and MSE loss (without the penalty term).
    """
    N, D = tx.shape
    A = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    w = np.linalg.solve(A, tx.T @ y)
    return w, _mse(y, tx, w)


def _sigmoid(z):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _logistic_loss(y, tx, w):
    """Negative log-likelihood for labels y in {0, 1}, averaged over samples."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Returns:
        (w, loss): last weight vector and its logistic loss.
    """
    w = initial_w.copy()
    N = len(y)
    for _ in range(max_iters):
        grad = tx.T @ (_sigmoid(tx @ w) - y) / N
        w = w - gamma * grad
    return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y in {0, 1}).

    The penalty is lambda_ * ||w||^2. The returned loss excludes the penalty.

    Returns:
        (w, loss): last weight vector and its logistic loss.
    """
    w = initial_w.copy()
    N = len(y)
    for _ in range(max_iters):
        grad = tx.T @ (_sigmoid(tx @ w) - y) / N + 2 * lambda_ * w
        w = w - gamma * grad
    return w, _logistic_loss(y, tx, w)
