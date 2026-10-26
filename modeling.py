"""Model selection and final ridge-model fitting for Project 1.

Only NumPy and the project's own ridge/preprocessing implementations are used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from implementations import ridge_regression
from preprocessing import DEFAULT_NOMINAL_FEATURES, preprocess_dataset


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Compute accuracy, balanced accuracy, precision, recall, and F1."""
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == -1) & (y_pred == -1)))
    false_positive = int(np.sum((y_true == -1) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == -1)))

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-15)
    accuracy = (true_positive + true_negative) / max(len(y_true), 1)
    balanced_accuracy = (recall + specificity) / 2

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC with average ranks for tied scores."""
    positive = y_true == 1
    positive_count = int(positive.sum())
    negative_count = len(y_true) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUC requires both positive and negative labels")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    _, first_indices, counts = np.unique(
        sorted_scores, return_index=True, return_counts=True
    )
    for first_index, count in zip(first_indices, counts):
        ranks[order[first_index:first_index + count]] = (
            first_index + 1 + first_index + count
        ) / 2

    rank_sum = ranks[positive].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def _scale_numeric_features(
    x_fit: np.ndarray,
    x_apply: np.ndarray,
    numeric_mask: np.ndarray,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize numeric feature columns; leave dummies/indicators unchanged."""
    fit_scaled = x_fit.copy()
    apply_scaled = x_apply.copy()
    fit_numeric = fit_scaled[:, :feature_count][:, numeric_mask]
    apply_numeric = apply_scaled[:, :feature_count][:, numeric_mask]

    means = np.mean(fit_numeric, axis=0)
    standard_deviations = np.std(fit_numeric, axis=0)
    standard_deviations[standard_deviations == 0] = 1.0

    fit_scaled[:, :feature_count][:, numeric_mask] = (
        fit_numeric - means
    ) / standard_deviations
    apply_scaled[:, :feature_count][:, numeric_mask] = (
        apply_numeric - means
    ) / standard_deviations
    return fit_scaled, apply_scaled


def _f1_threshold_summary(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    """Find the exact score threshold with maximum F1 and summarize it."""
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    cumulative_true_positives = np.cumsum(sorted_labels == 1)
    cumulative_false_positives = np.cumsum(sorted_labels == -1)
    group_ends = np.r_[
        np.flatnonzero(np.diff(sorted_scores) != 0),
        len(sorted_scores) - 1,
    ]

    true_positives = cumulative_true_positives[group_ends]
    false_positives = cumulative_false_positives[group_ends]
    false_negatives = int(np.sum(labels == 1)) - true_positives
    true_negatives = int(np.sum(labels == -1)) - false_positives
    thresholds = sorted_scores[group_ends]

    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / max(int(np.sum(labels == 1)), 1)
    specificity = true_negatives / max(int(np.sum(labels == -1)), 1)
    f1_values = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    best_index = int(np.argmax(f1_values))

    return {
        "threshold": float(thresholds[best_index]),
        "accuracy": float(
            (true_positives[best_index] + true_negatives[best_index]) / len(labels)
        ),
        "balanced_accuracy": float(
            (recall[best_index] + specificity[best_index]) / 2
        ),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1_values[best_index]),
        "true_positive": int(true_positives[best_index]),
        "false_positive": int(false_positives[best_index]),
        "false_negative": int(false_negatives[best_index]),
        "true_negative": int(true_negatives[best_index]),
    }


def select_ridge_by_oof_f1(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    feature_names: list[str],
    *,
    missing_threshold: float = 0.05,
    lambdas: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
    fold_count: int = 3,
    random_state: int = 123,
    nominal_features: tuple[str, ...] = DEFAULT_NOMINAL_FEATURES,
) -> dict[str, Any]:
    """Select ridge regularization and score threshold by out-of-fold F1.

    Feature filtering, encoding, imputation, and numeric scaling are refitted
    within each fold. Returned maximum F1 is a selection estimate, not an
    unbiased generalization score, because the same OOF labels select lambda
    and threshold.
    """
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if len(lambdas) == 0 or any(lambda_value < 0 for lambda_value in lambdas):
        raise ValueError("lambdas must contain nonnegative values")

    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train)
    train_ids = np.asarray(train_ids)
    fold_generator = np.random.default_rng(random_state)
    fold_assignment = np.full(len(y_train), -1, dtype=int)
    for label in np.unique(y_train):
        label_indices = np.flatnonzero(y_train == label)
        fold_generator.shuffle(label_indices)
        for fold_index, indices in enumerate(np.array_split(label_indices, fold_count)):
            fold_assignment[indices] = fold_index

    oof_scores = np.full((len(y_train), len(lambdas)), np.nan)
    fold_reports = []

    for fold_index in range(fold_count):
        validation_indices = np.flatnonzero(fold_assignment == fold_index)
        fit_indices = np.flatnonzero(fold_assignment != fold_index)
        fold_data = preprocess_dataset(
            x_train[fit_indices],
            x_train[validation_indices],
            y_train[fit_indices],
            train_ids[fit_indices],
            train_ids[validation_indices],
            feature_names,
            missing_threshold=missing_threshold,
            validation_fraction=0,
            random_state=random_state + fold_index,
            nominal_features=nominal_features,
        )

        x_fit = fold_data["x_train_processed"]
        x_validation = fold_data["x_test_processed"]
        y_fit = y_train[fit_indices]
        selected_feature_count = len(fold_data["kept_feature_names"])
        numeric_mask = np.asarray(
            [
                source_name in fold_data["numeric_feature_names"]
                for source_name, keep in zip(
                    fold_data["encoded_source_names"], fold_data["keep_features"]
                )
                if keep
            ],
            dtype=bool,
        )
        x_fit, x_validation = _scale_numeric_features(
            x_fit, x_validation, numeric_mask, selected_feature_count
        )
        tx_fit = np.column_stack((np.ones(len(x_fit)), x_fit))
        tx_validation = np.column_stack((np.ones(len(x_validation)), x_validation))

        gram_matrix = tx_fit.T @ tx_fit
        cross_product = tx_fit.T @ y_fit
        for lambda_index, lambda_value in enumerate(lambdas):
            regularized_matrix = gram_matrix + (
                2 * len(y_fit) * lambda_value * np.eye(tx_fit.shape[1])
            )
            weights = np.linalg.solve(regularized_matrix, cross_product)
            oof_scores[validation_indices, lambda_index] = tx_validation @ weights

        fold_reports.append(
            {
                "fold": fold_index + 1,
                "fit_rows": len(fit_indices),
                "validation_rows": len(validation_indices),
                "features": len(fold_data["kept_feature_names"]),
            }
        )

    lambda_results = []
    for lambda_index, lambda_value in enumerate(lambdas):
        scores = oof_scores[:, lambda_index]
        predictions_at_zero = np.where(scores >= 0, 1, -1)
        threshold_result = _f1_threshold_summary(y_train, scores)
        lambda_results.append(
            {
                "lambda": float(lambda_value),
                "auc": binary_auc(y_train, scores),
                "metrics_at_zero": classification_metrics(y_train, predictions_at_zero),
                "best_f1": threshold_result,
            }
        )

    best_result = max(lambda_results, key=lambda result: result["best_f1"]["f1"])
    return {
        "best_lambda": best_result["lambda"],
        "best_threshold": best_result["best_f1"]["threshold"],
        "best_f1_result": best_result["best_f1"],
        "best_auc": best_result["auc"],
        "lambda_results": lambda_results,
        "fold_reports": fold_reports,
        "out_of_fold_scores": oof_scores,
        "lambdas": tuple(float(value) for value in lambdas),
    }


def fit_final_ridge(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    feature_names: list[str],
    *,
    lambda_value: float,
    threshold: float,
    missing_threshold: float = 0.05,
    nominal_features: tuple[str, ...] = DEFAULT_NOMINAL_FEATURES,
) -> dict[str, Any]:
    """Refit preprocessing/model on all labeled rows and predict test labels."""
    final_data = preprocess_dataset(
        x_train,
        x_test,
        y_train,
        train_ids,
        test_ids,
        feature_names,
        missing_threshold=missing_threshold,
        validation_fraction=0,
        nominal_features=nominal_features,
    )
    x_fit = final_data["x_train_processed"]
    x_test_processed = final_data["x_test_processed"]
    selected_feature_count = len(final_data["kept_feature_names"])
    numeric_mask = np.asarray(
        [
            source_name in final_data["numeric_feature_names"]
            for source_name, keep in zip(
                final_data["encoded_source_names"], final_data["keep_features"]
            )
            if keep
        ],
        dtype=bool,
    )
    x_fit, x_test_processed = _scale_numeric_features(
        x_fit, x_test_processed, numeric_mask, selected_feature_count
    )
    tx_fit = np.column_stack((np.ones(len(x_fit)), x_fit))
    tx_test = np.column_stack((np.ones(len(x_test_processed)), x_test_processed))
    weights, training_loss = ridge_regression(y_train, tx_fit, lambda_value)
    scores = tx_test @ weights
    predictions = np.where(scores >= threshold, 1, -1)

    return {
        "preprocessing": final_data,
        "weights": weights,
        "training_loss": training_loss,
        "scores": scores,
        "predictions": predictions,
        "positive_predictions": int(np.sum(predictions == 1)),
    }
