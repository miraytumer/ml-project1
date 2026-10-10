"""
Preprocessing for the BRFSS heart-disease data (CS-433 Project 1).

Goal: turn the raw 321-column survey matrix into a clean numeric matrix
(49 columns with the CORE set) that a linear model can learn from.

USAGE
    x_train, x_test, y_train, train_ids, test_ids, names = load_data("data/")
    Xc_train = clean(x_train, names)        # fixed rules, same for everyone
    p        = fit(Xc_train)                # statistics from TRAINING data only
    tX_train = transform(Xc_train, p)       # final matrix (bias column first)
    tX_test  = transform(clean(x_test, names), p)

STAGE 1 - clean(): fixed codebook rules, nothing learned from data
    1. Keep only the CORE features (32 of 321). We dropped:
       survey admin/weights (describe the phone call, not the person),
       leakage (questions only asked to heart patients, e.g. HAREHAB1),
       features >50% missing (optional survey modules),
       duplicates (kept one version per concept, e.g. _AGE80 not _AGEG5YR),
       weak features (eta^2 < 0.01: heart-disease rate barely differs between answers).
    2. Fix special codes. The survey stores "don't know / refused" as numbers
       (7, 9, 77, 99, 99900...). Left as numbers, the model would read them as
       extreme answers, so they become NaN (CODES_TO_NAN).
    3. Two codes are real answers, not unknowns:
       PHYSHLTH 88 = "no bad days" -> 0
       BPMEDS missing = only asked if high blood pressure -> "no" (2)
       (checked in data: all missing BPMEDS have no high BP)

STAGE 2 - fit() learns, transform() applies (prevents data leakage)
    fit() computes statistics on training data only; transform() applies the
    SAME statistics to train, validation and test. If validation/test rows
    influenced these statistics, our scores would be too optimistic.
    In cross-validation, fit() is re-run on each fold's training part.

    Steps (done in _expand, then standardized in transform):
    1. Clip heavy tails (CLIP) at the training 99th percentile:
       max values like 150 fruit servings/day are data-entry errors.
    2. Missing indicators (MISSING_IND): 0/1 column "was missing", added BEFORE
       imputing, because here missingness means something
       (_HCVU651 missing = age 65+, _RFCHOL missing = never checked, _INCOMG = refused).
    3. One-hot encode unordered categories (ONE_HOT), e.g. EMPLOY1 7 ("retired")
       is not "more" than 6 ("student"). Missing -> all zeros.
       First category dropped: with a bias column the full set is redundant
       and makes least squares singular.
    4. Fill remaining NaN with the training median (robust to outliers).
    5. Standardize every column (mean 0, std 1): features have very different
       scales (age 18-80, binary 1/2, drinks 0-2800); this makes gradient
       descent stable and regularization fair across features.
    6. Add a bias column of ones (lets the model learn the base rate, ~8.8%).

All choices (feature list, thresholds) are in the lists below so they can be
changed for ablation experiments without touching the logic.
"""



import numpy as np

CORE = ["_RFHYPE5","BPMEDS","_RFCHOL","_CHOLCHK","_ASTHMS1","_DRDXAR1","_HCVU651",
        "CVDSTRK3","CHCCOPD1","DIABETE3","GENHLTH","PHYSHLTH","DIFFWALK","USEEQUIP","QLACTLM2",
        "_AGE80","SEX","_RACEGR3","_EDUCAG","_INCOMG","_CHLDCNT","EMPLOY1","_BMI5",
        "_SMOKER3","_DRNKWEK","_RFBING5","_FRUTSUM","_VEGESUM","_PACAT1","_PASTRNG",
        "_LMTACT1","_LMTWRK1"]

CODES_TO_NAN = {
    **{n: [9] for n in ["_RFHYPE5","_RFCHOL","_CHOLCHK","_ASTHMS1","_HCVU651","_RACEGR3",
                        "_EDUCAG","_INCOMG","_CHLDCNT","_SMOKER3","_RFBING5","_PACAT1",
                        "_PASTRNG","_LMTACT1","_LMTWRK1","EMPLOY1"]},
    **{n: [7, 9] for n in ["BPMEDS","CVDSTRK3","CHCCOPD1","DIABETE3","GENHLTH",
                           "DIFFWALK","USEEQUIP","QLACTLM2"]},
    "_DRNKWEK": [99900],
    "PHYSHLTH": [77, 99],
}

ONE_HOT = ["EMPLOY1", "DIABETE3", "_RACEGR3", "_LMTACT1", "_LMTWRK1"]


def clean(X, feature_names, features=CORE):
    """Select features and apply fixed codebook rules."""
    idx = {n: i for i, n in enumerate(feature_names)}
    Xc = X[:, [idx[n] for n in features]].copy()
    col = {n: j for j, n in enumerate(features)}

    if "BPMEDS" in col:                      # only asked if high BP -> missing means "no"
        j = col["BPMEDS"]; Xc[np.isnan(Xc[:, j]), j] = 2
    if "PHYSHLTH" in col:                    # 88 = "none" = 0 days
        j = col["PHYSHLTH"]; Xc[Xc[:, j] == 88, j] = 0
    for n, codes in CODES_TO_NAN.items():
        if n in col:
            j = col[n]; Xc[np.isin(Xc[:, j], codes), j] = np.nan
    return Xc

MISSING_IND = ["_HCVU651", "_RFCHOL", "_INCOMG"]   # missingness carries information
CLIP = ["_DRNKWEK", "_FRUTSUM", "_VEGESUM", "_BMI5"]


def _expand(Xc, p, features):
    """Clip, add missing indicators, one-hot categoricals, median-impute numerics."""
    col = {n: j for j, n in enumerate(features)}
    Xc = Xc.copy()
    for n in CLIP:                                   # NaN stays NaN under np.minimum
        j = col[n]; Xc[:, j] = np.minimum(Xc[:, j], p["clip"][n])

    indicators = [np.isnan(Xc[:, col[n]]).astype(float) for n in MISSING_IND]

    parts = []
    for n in features:
        j = col[n]
        if n in ONE_HOT:                             # missing -> all zeros; first category dropped
            parts += [(Xc[:, j] == v).astype(float) for v in p["cats"][n][1:]]
        else:
            c = Xc[:, j]
            parts.append(np.where(np.isnan(c), p["median"][j], c))
    return np.column_stack(parts + indicators)


def fit(Xc, features=CORE):
    """Learn all statistics from TRAINING data only."""
    col = {n: j for j, n in enumerate(features)}
    p = {
        "clip":   {n: np.nanpercentile(Xc[:, col[n]], 99) for n in CLIP},
        "median": np.nanmedian(Xc, axis=0),
        "cats":   {n: np.unique(Xc[~np.isnan(Xc[:, col[n]]), col[n]]) for n in ONE_HOT},
    }
    Z = _expand(Xc, p, features)
    p["mean"], p["std"] = Z.mean(axis=0), Z.std(axis=0)
    p["std"][p["std"] == 0] = 1.0
    return p


def transform(Xc, p, features=CORE):
    """Apply learned statistics, standardize, add bias column."""
    Z = (_expand(Xc, p, features) - p["mean"]) / p["std"]
    return np.column_stack([np.ones(len(Z)), Z])