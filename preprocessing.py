"""Preprocessing pipeline for the BRFSS health dataset.

This module keeps the notebook focused on experiments while centralizing the
reusable data preparation steps. It uses only Python's standard library and
NumPy.

Pipeline
--------
1. Create a reproducible stratified train/validation split.
2. Recode the verified survey special values and normalize weight units.
3. One-hot encode configured nominal features using levels learned on train.
4. Select features using training-set missingness and constant-column checks.
5. Impute from the training split and append missingness indicators.

All learned transformations are fitted on the training partition, then reused
unchanged for validation and test data.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Features represented as categories instead of numeric magnitudes. Edit this
# list independently of `missing_threshold` in preprocess_dataset().
DEFAULT_NOMINAL_FEATURES = (
    "HLTHPLN1", "PERSDOC2", "MEDCOST", "BPHIGH4", "BLOODCHO", "CVDSTRK3",
    "ASTHMA3", "CHCSCNCR", "CHCOCNCR", "CHCCOPD1", "HAVARTH3", "ADDEPEV2",
    "CHCKIDNY", "DIABETE3", "SEX", "MARITAL", "RENTHOM1", "VETERAN3",
    "EMPLOY1", "INTERNET", "PREGNANT", "QLACTLM2", "USEEQUIP", "BLIND",
    "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON", "SMOKE100", "USENOW3",
    "DRNKANY5", "_PRACE1", "_MRACE1", "_RACE", "_RACEG21", "_RACEGR3",
    "_RACE_G1", "_RFHLTH", "_RFHYPE5", "_CHOLCHK", "_LTASTH1", "_CASTHM1",
    "_ASTHMS1", "_DRDXAR1", "_HISPANC", "_RFBMI5", "_SMOKER3", "_RFSMOK3",
    "_RFBING5", "_RFDRHV5", "_FRTLT1", "_VEGLT1", "_TOTINDA", "_PACAT1",
    "_PAINDX1", "_PA150R2", "_PA300R2", "_PA30021", "_PASTRNG", "_PAREC1",
    "_PASTAE1", "_LMTACT1", "_LMTWRK1", "_LMTSCL1", "_RFSEAT2", "_RFSEAT3",
    "_FLSHOT6", "_PNEUMO2", "_AIDTST3",
)

# These columns describe survey administration/design rather than respondent
# characteristics for the initial predictive baseline.
ADMINISTRATIVE_FEATURES = frozenset(
    {
        "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO",
        "QSTVER", "QSTLANG", "MSCODE", "_STATE", "_PSU", "_STSTR", "_RAWRAKE",
        "_WT2RAKE", "_CHISPNC", "_CPRACE", "_CLLCPWT", "_DUALUSE", "_DUALCOR",
        "_LLCPWT",
    }
)

# Valid numeric/category codes that overlap the generic unknown/refused codes.
VALID_VALUES_BY_FEATURE = {
    "PHYSHLTH": set(range(0, 31)) | {88},
    "MENTHLTH": set(range(0, 31)) | {88},
    "CHECKUP1": {8},
    "EMPLOY1": set(range(1, 9)),
    "CHILDREN": set(range(1, 89)),
    "_PRACE1": {7, 8},
    "_MRACE1": {7},
    "_RACE": {7, 8},
    "_AGEG5YR": set(range(1, 14)),
    "DROCDY3_": set(range(0, 900)),
    "INCOME2": set(range(1, 9)),
    "_AGE80": set(range(18, 100)),
    "WEIGHT2": set(range(50, 1000)) | set(range(9000, 9999)),
    "HTIN4": set(range(36, 96)),
}

# Additional feature-specific sentinels confirmed in the project codebook.
ADDITIONAL_MISSING_CODES_BY_FEATURE = {
    "PHYSHLTH": {77, 99},
    "MENTHLTH": {77, 99},
    "CHILDREN": {99},
    "_PRACE1": {77, 99},
    "_MRACE1": {77, 99},
    "_RACE": {9},
    "_AGEG5YR": {14},
    "DROCDY3_": {900},
    "INCOME2": {77, 99},
    "WEIGHT2": {7777, 9999},
}

ZERO_CODES_BY_FEATURE = {
    "PHYSHLTH": 88,
    "MENTHLTH": 88,
    "CHILDREN": 88,
}


def _build_special_missing_codes(feature_names: list[str]) -> dict[str, set[int]]:
    """Build feature-specific missing-code mappings from verified rules."""
    mappings = {
        name: {
            code
            for code in (7, 9)
            if code not in VALID_VALUES_BY_FEATURE.get(name, set())
        }
        for name in feature_names
        if name not in ADMINISTRATIVE_FEATURES
    }
    mappings = {name: codes for name, codes in mappings.items() if codes}

    for name, codes in ADDITIONAL_MISSING_CODES_BY_FEATURE.items():
        if name in feature_names:
            mappings.setdefault(name, set()).update(codes)
    return mappings


def _recode_survey_values(
    matrix: np.ndarray,
    feature_names: list[str],
    special_missing_codes_by_feature: dict[str, set[int]],
) -> np.ndarray:
    """Apply verified special-code replacements and normalize WEIGHT2."""
    recoded = matrix.copy()
    feature_index = {name: index for index, name in enumerate(feature_names)}

    for name, codes in special_missing_codes_by_feature.items():
        if name in feature_index:
            column_index = feature_index[name]
            is_special_missing = np.isin(recoded[:, column_index], tuple(codes))
            recoded[is_special_missing, column_index] = np.nan

    for name, code in ZERO_CODES_BY_FEATURE.items():
        if name in feature_index:
            column_index = feature_index[name]
            recoded[recoded[:, column_index] == code, column_index] = 0.0

    if "WEIGHT2" in feature_index:
        column_index = feature_index["WEIGHT2"]
        weights = recoded[:, column_index]
        pounds_mask = (weights >= 50) & (weights <= 999)
        kilograms_mask = (weights >= 9000) & (weights <= 9998)
        weights[pounds_mask] /= 2.20462
        weights[kilograms_mask] -= 9000

    return recoded


def _encode_nominal_features(
    matrix: np.ndarray,
    feature_names: list[str],
    nominal_feature_names: list[str],
    nominal_levels: dict[str, np.ndarray],
    numeric_column_indices: list[int],
) -> np.ndarray:
    """Encode nominal columns, preserving missing rows as NaN dummy rows."""
    encoded_parts = [matrix[:, numeric_column_indices]]
    feature_index = {name: index for index, name in enumerate(feature_names)}

    for name in nominal_feature_names:
        values = matrix[:, feature_index[name]]
        levels = nominal_levels[name]
        if levels.size == 0:
            continue
        encoded = np.column_stack([values == level for level in levels]).astype(float)
        encoded[~np.isfinite(values), :] = np.nan
        encoded_parts.append(encoded)

    return np.column_stack(encoded_parts)


def _stratified_split(
    labels: np.ndarray,
    validation_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic, class-stratified fit and validation indices."""
    if validation_fraction == 0:
        return np.arange(labels.size, dtype=int), np.empty(0, dtype=int)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be 0 or strictly between 0 and 1")

    generator = np.random.default_rng(random_state)
    fit_indices: list[int] = []
    validation_indices: list[int] = []
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        if label_indices.size < 2:
            raise ValueError(f"Each class needs at least two samples; class {label!r} does not")
        generator.shuffle(label_indices)
        validation_count = max(1, int(round(validation_fraction * label_indices.size)))
        validation_count = min(validation_count, label_indices.size - 1)
        validation_indices.extend(label_indices[:validation_count])
        fit_indices.extend(label_indices[validation_count:])

    return (
        np.sort(np.asarray(fit_indices, dtype=int)),
        np.sort(np.asarray(validation_indices, dtype=int)),
    )


def preprocess_dataset(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    feature_names: list[str],
    *,
    missing_threshold: float = 0.05,
    validation_fraction: float = 0.20,
    random_state: int = 42,
    nominal_features: tuple[str, ...] = DEFAULT_NOMINAL_FEATURES,
) -> dict[str, Any]:
    """Prepare split, recoded, one-hot encoded and imputed model matrices.

    Parameters
    ----------
    x_train, x_test:
        Raw numeric predictor arrays, with ID columns already removed.
    y_train:
        Training labels. A stratified validation subset is made from these.
    train_ids, test_ids:
        Row IDs corresponding to the input arrays; they are split/preserved in
        parallel with the labels and predictors.
    feature_names:
        CSV feature names in the same order as columns in ``x_train``/``x_test``.
    missing_threshold:
        Maximum fraction of missing fit-split values allowed for an encoded
        feature. This is deliberately independent of the nominal feature list.
    validation_fraction, random_state:
        Controls the reproducible stratified split. Use zero to fit all labeled
        rows and return an empty validation partition, e.g. for final refitting.
    nominal_features:
        Original feature names to one-hot encode. Levels are learned from the
        fit split only. Features not present in ``feature_names`` are ignored.

    Returns
    -------
    dict[str, Any]
        A dictionary containing intermediate arrays/metadata for audits and the
        processed train, validation, and test matrices.
    """
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    y_train = np.asarray(y_train)
    train_ids = np.asarray(train_ids)
    test_ids = np.asarray(test_ids)
    feature_names = list(feature_names)

    if not 0 <= missing_threshold <= 1:
        raise ValueError("missing_threshold must be between 0 and 1")
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("x_train and x_test must be two-dimensional arrays")
    if x_train.shape[1] != len(feature_names) or x_test.shape[1] != len(feature_names):
        raise ValueError("Feature-name count must match train and test column counts")
    if x_train.shape[0] != y_train.size or x_train.shape[0] != train_ids.size:
        raise ValueError("Training predictors, labels, and IDs must have matching row counts")
    if x_test.shape[0] != test_ids.size:
        raise ValueError("Test predictors and IDs must have matching row counts")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be unique")

    # 1. Split before fitting any data-dependent transformations.
    train_indices, validation_indices = _stratified_split(
        y_train, validation_fraction, random_state
    )
    x_fit_raw = x_train[train_indices]
    y_fit = y_train[train_indices]
    train_ids_fit = train_ids[train_indices]
    x_validation_raw = x_train[validation_indices]
    y_validation = y_train[validation_indices]
    validation_ids = train_ids[validation_indices]

    # 2. Apply the confirmed codebook mappings consistently to every partition.
    special_missing_codes_by_feature = _build_special_missing_codes(feature_names)
    x_fit_recoded = _recode_survey_values(
        x_fit_raw, feature_names, special_missing_codes_by_feature
    )
    x_validation_recoded = _recode_survey_values(
        x_validation_raw, feature_names, special_missing_codes_by_feature
    )
    x_test_recoded = _recode_survey_values(
        x_test, feature_names, special_missing_codes_by_feature
    )

    # 3. Fit the nominal schema on fit data; reuse it unchanged elsewhere.
    nominal_feature_names = [
        name
        for name in nominal_features
        if name in feature_names and name not in ADMINISTRATIVE_FEATURES
    ]
    nominal_column_indices = [feature_names.index(name) for name in nominal_feature_names]
    numeric_column_indices = [
        index for index, name in enumerate(feature_names) if name not in nominal_feature_names
    ]
    numeric_feature_names = [feature_names[index] for index in numeric_column_indices]
    nominal_levels = {}
    for name in nominal_feature_names:
        column_index = feature_names.index(name)
        observed = x_fit_recoded[
            np.isfinite(x_fit_recoded[:, column_index]), column_index
        ]
        nominal_levels[name] = np.unique(observed)

    nominal_encoded_names = [
        f"{name}={level:g}"
        for name in nominal_feature_names
        for level in nominal_levels[name]
    ]
    x_fit_encoded = _encode_nominal_features(
        x_fit_recoded,
        feature_names,
        nominal_feature_names,
        nominal_levels,
        numeric_column_indices,
    )
    x_validation_encoded = _encode_nominal_features(
        x_validation_recoded,
        feature_names,
        nominal_feature_names,
        nominal_levels,
        numeric_column_indices,
    )
    x_test_encoded = _encode_nominal_features(
        x_test_recoded,
        feature_names,
        nominal_feature_names,
        nominal_levels,
        numeric_column_indices,
    )
    encoded_feature_names = numeric_feature_names + nominal_encoded_names
    encoded_source_names = numeric_feature_names + [
        name for name in nominal_feature_names for _ in nominal_levels[name]
    ]

    # 4. Select encoded columns using fit-split missingness and variance only.
    train_missing_rates = np.mean(~np.isfinite(x_fit_encoded), axis=0)
    train_constant = np.asarray(
        [
            np.unique(
                x_fit_encoded[np.isfinite(x_fit_encoded[:, index]), index]
            ).size <= 1
            for index in range(x_fit_encoded.shape[1])
        ],
        dtype=bool,
    )
    keep_features = np.asarray(
        [name not in ADMINISTRATIVE_FEATURES for name in encoded_source_names],
        dtype=bool,
    ) & (train_missing_rates <= missing_threshold) & ~train_constant
    kept_feature_names = [
        name for name, keep in zip(encoded_feature_names, keep_features) if keep
    ]
    x_fit_selected = x_fit_encoded[:, keep_features]
    x_validation_selected = x_validation_encoded[:, keep_features]
    x_test_selected = x_test_encoded[:, keep_features]

    # 5. Learn medians and indicator schema on fit data, then transform all sets.
    fit_missing_mask = ~np.isfinite(x_fit_selected)
    finite_fit_values = np.where(np.isfinite(x_fit_selected), x_fit_selected, np.nan)
    imputation_values = np.nanmedian(finite_fit_values, axis=0)
    missing_indicator_columns = np.any(fit_missing_mask, axis=0)
    missing_indicator_names = [
        f"{name}_missing"
        for name, has_missing in zip(kept_feature_names, missing_indicator_columns)
        if has_missing
    ]

    def impute_and_add_indicators(matrix: np.ndarray) -> np.ndarray:
        missing_mask = ~np.isfinite(matrix)
        imputed = matrix.copy()
        missing_rows, missing_columns = np.where(missing_mask)
        imputed[missing_rows, missing_columns] = imputation_values[missing_columns]
        if np.any(missing_indicator_columns):
            indicators = missing_mask[:, missing_indicator_columns].astype(float)
            imputed = np.column_stack((imputed, indicators))
        return imputed

    x_train_processed = impute_and_add_indicators(x_fit_selected)
    x_validation_processed = impute_and_add_indicators(x_validation_selected)
    x_test_processed = impute_and_add_indicators(x_test_selected)
    processed_feature_names = kept_feature_names + missing_indicator_names

    return {
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "x_fit_raw": x_fit_raw,
        "y_fit": y_fit,
        "train_ids_fit": train_ids_fit,
        "x_validation_raw": x_validation_raw,
        "y_validation": y_validation,
        "validation_ids": validation_ids,
        "feature_names": feature_names,
        "administrative_features": set(ADMINISTRATIVE_FEATURES),
        "valid_values_by_feature": VALID_VALUES_BY_FEATURE,
        "special_missing_codes_by_feature": special_missing_codes_by_feature,
        "zero_code_by_feature": ZERO_CODES_BY_FEATURE,
        "recode_survey_values": lambda matrix: _recode_survey_values(
            matrix, feature_names, special_missing_codes_by_feature
        ),
        "x_fit_recoded": x_fit_recoded,
        "x_validation_recoded": x_validation_recoded,
        "x_test_recoded": x_test_recoded,
        "nominal_feature_names": nominal_feature_names,
        "nominal_column_indices": nominal_column_indices,
        "numeric_column_indices": numeric_column_indices,
        "numeric_feature_names": numeric_feature_names,
        "nominal_levels": nominal_levels,
        "nominal_encoded_names": nominal_encoded_names,
        "x_fit_encoded": x_fit_encoded,
        "x_validation_encoded": x_validation_encoded,
        "x_test_encoded": x_test_encoded,
        "encoded_feature_names": encoded_feature_names,
        "encoded_source_names": encoded_source_names,
        "train_missing_rates": train_missing_rates,
        "train_constant": train_constant,
        "keep_features": keep_features,
        "kept_feature_names": kept_feature_names,
        "x_fit_selected": x_fit_selected,
        "x_validation_selected": x_validation_selected,
        "x_test_selected": x_test_selected,
        "fit_missing_mask": fit_missing_mask,
        "imputation_values": imputation_values,
        "missing_indicator_columns": missing_indicator_columns,
        "missing_indicator_names": missing_indicator_names,
        "x_train_processed": x_train_processed,
        "x_validation_processed": x_validation_processed,
        "x_test_processed": x_test_processed,
        "processed_feature_names": processed_feature_names,
    }
