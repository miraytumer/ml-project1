"""
Preprocessing for the BRFSS heart-disease data (CS-433 Project 1).

    x_train, x_test, y_train, train_ids, test_ids, names = load_data("dataset/")
    tX_train, tX_test, p = prepare(x_train, x_test, names)

clean()      fixed rules (same for everyone): drop admin / leakage / redundant columns, fill
             missing values whose meaning is known, survey codes -> NaN ("none" 88 -> 0).
fit()        DECIDES what to do, from the TRAINING data only: which columns to keep, which to
             one-hot, clip values, medians, means and stds for standardization.
transform()  DOES it, on any data (train, validation, test), and adds a bias column.
The split guarantees that validation/test data never influence the statistics.

How the drop lists were built (see EDA):
  ADMIN      survey bookkeeping (dates, ids, phone screening, sampling weights).
  LEAKAGE    questions that reveal the label: everyone who answered HAREHAB1 (cardiac rehab)
             has heart disease (checked: 100%).
  REDUNDANT  another version of a concept we keep. The CDC provides cleaned "_" variables
             computed from the raw answers; we keep one version per concept (usually the "_"
             one, except GENHLTH whose 1-5 scale is richer than _RFHLTH). Pairs were found from
             the codebook and confirmed in the data by correlation (e.g. _LMTACT1=3 and
             _LMTWRK1=3 were identical, which broke least squares).
"""

import os
import numpy as np
from helpers import load_csv_data

# ---------------- decisions (from the EDA) ----------------

# survey bookkeeping: describes the phone call, not the person
ADMIN = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "CTELENUM",
    "PVTRESD1",
    "COLGHOUS",
    "STATERES",
    "CELLFON3",
    "LADULT",
    "NUMADULT",
    "NUMMEN",
    "NUMWOMEN",
    "CTELNUM1",
    "CELLFON2",
    "CADULT",
    "PVTRESD2",
    "CCLGHOUS",
    "CSTATE",
    "LANDLINE",
    "HHADULT",
    "QSTVER",
    "QSTLANG",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_DUALUSE",
    "_DUALCOR",
    "_LLCPWT",
    "_CHISPNC",
    "_CRACE1",
    "_CPRACE",
]

# reveal the label: HAREHAB1 (cardiac rehab) is answered only by heart patients (100% positive)
LEAKAGE = ["HAREHAB1", "CVDINFR4", "CVDCRHD4"]

# another version of a concept we already keep (we keep the cleaned "_" variable)
REDUNDANT = [
    "_AGEG5YR",
    "_AGE_G",
    "_AGE65YR",
    "FC60_",
    "MAXVO2_",  # age
    "_RFHLTH",
    "BPHIGH4",
    "TOLDHI2",
    "CHOLCHK",
    "DIABAGE2",  # health
    "HAVARTH3",
    "_DRDXAR1",
    "_LMTWRK1",
    "_LMTSCL1",  # arthritis
    "_PRACE1",
    "_MRACE1",
    "_HISPANC",
    "_RACE",
    "_RACEG21",
    "_RACE_G1",  # race
    "HTIN4",
    "HTM4",
    "WTKG3",
    "_BMI5CAT",
    "_RFBMI5",
    "WEIGHT2",
    "HEIGHT3",  # BMI
    "_RFSMOK3",
    "SMOKE100",
    "SMOKDAY2",  # smoking
    "_RFDRHV5",
    "ALCDAY5",
    "AVEDRNK2",
    "MAXDRNKS",
    "DRNK3GE5",  # alcohol
    "INCOME2",
    "EDUCA",
    "CHILDREN",  # demographics
    "_LTASTH1",
    "_CASTHM1",
    "ASTHMA3",
    "ASTHNOW",  # asthma
    "_MISFRTN",
    "_MISVEGN",
    "_FRTRESP",
    "_VEGRESP",
    "_FRTLT1",
    "_VEGLT1",
    "_FRT16",
    "_VEG23",
    "_FRUITEX",
    "_VEGETEX",
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    "_TOTINDA",
    "_MINAC11",
    "_MINAC21",
    "_PAINDX1",
    "_PA150R2",
    "_PA300R2",
    "_PA30021",
    "_PAREC1",
    "_PASTAE1",
    "EXERANY2",
    "EXRACT11",
    "EXRACT21",
    "STRENGTH",
    "EXEROFT1",
    "EXEROFT2",
    "EXERHMM1",
    "EXERHMM2",  # odd encodings
    "_AIDTST3",
    "BLOODCHO",  # exact copies of HIVTST6 and _CHOLCHK=3 (|r| > 0.99)
]

# survey codes: a value from SPECIAL is a code only if it lies above all normal values
SPECIAL = {7, 8, 9, 77, 88, 99, 777, 888, 999, 7777, 8888, 9999, 99900}
ZERO_CODES = {88, 888}  # "none" -> 0
CODE_EXCEPTIONS = {
    "EMPLOY1": [9],  # 7 = retired, 8 = unable to work are real answers
    "CHECKUP1": [7, 9],
}  # 8 = never had a checkup is a real answer

STRUCTURAL_FILLS = {
    "BPMEDS": 2,  # only asked if high blood pressure -> missing = "no"
    "POORHLTH": 0,
}  # only asked if bad days > 0 -> missing = 0 days
MISSING_INDICATORS = ["_HCVU651", "_RFCHOL", "_INCOMG"]

MAX_MISSING = (
    0.5  # drop a column if >50% missing: in the EDA features were either nearly
)
# complete or mostly missing (optional survey modules), with a gap between
MAX_LEVELS = 10  # 3 to 10 distinct values = a category or short scale -> one-hot encode
CLIP_PCT = 99  # clip continuous columns at the 99th percentile: the max values were
# data-entry errors (e.g. 150 fruit servings a day)
CONTINUOUS_LEVELS = (
    16  # only columns with >= 16 distinct values count as continuous (clipped)
)


# ---------------- loading ----------------


def load_data(data_dir="data/"):
    """Load the CSVs once, then from the cache data/cache.npz (about a second)."""
    cache = os.path.join(data_dir, "cache.npz")
    if not os.path.exists(cache):
        x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_dir)
        with open(os.path.join(data_dir, "x_train.csv")) as f:
            names = np.array(f.readline().strip().split(",")[1:])
        np.savez(
            cache,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            train_ids=train_ids,
            test_ids=test_ids,
            feature_names=names,
        )
    d = np.load(cache)
    return (
        d["x_train"],
        d["x_test"],
        d["y_train"],
        d["train_ids"],
        d["test_ids"],
        [str(n) for n in d["feature_names"]],
    )


# ---------------- stage 1: fixed rules ----------------


def detect_codes(col):
    """Values of SPECIAL that lie above every normal value of the column."""
    values = np.unique(col[~np.isnan(col)])
    normal = values[~np.isin(values, list(SPECIAL))]
    return [v for v in values if v in SPECIAL and len(normal) and v > normal.max()]


def clean(X, names):
    """Drop admin / leakage / redundant columns, structural fills, codes -> NaN (or 0)."""
    drop = set(ADMIN) | set(LEAKAGE) | set(REDUNDANT)
    kept = [n for n in names if n not in drop]
    Xc = X[:, [names.index(n) for n in kept]].copy()
    for j, n in enumerate(kept):
        x = Xc[:, j]  # view: edits change Xc
        if n in STRUCTURAL_FILLS:
            x[np.isnan(x)] = STRUCTURAL_FILLS[n]
        for c in CODE_EXCEPTIONS.get(n, detect_codes(x)):
            x[x == c] = 0.0 if c in ZERO_CODES else np.nan
    return Xc, kept


# ---------------- stage 2: learn on train, apply anywhere ----------------


def _build(Xc, names, p):
    """One-hot, clip, median-fill and missing indicators, using the statistics in p."""
    c = {n: j for j, n in enumerate(names)}
    blocks, cols = [], []
    for n in p["cols"]:
        x = Xc[:, c[n]]
        if n in p["onehot"]:  # missing -> all zeros
            blocks += [(x == v).astype(float) for v in p["onehot"][n]]
            cols += [f"{n}={v:g}" for v in p["onehot"][n]]
        else:
            if n in p["clip"]:
                x = np.minimum(x, p["clip"][n])
            blocks.append(np.where(np.isnan(x), p["median"][n], x))
            cols.append(n)
    for n in p["indicators"]:
        blocks.append(np.isnan(Xc[:, c[n]]).astype(float))
        cols.append(f"{n}_missing")
    return np.column_stack(blocks), cols


def fit(Xc, names):
    """Decide everything from the cleaned TRAINING data; transform() applies it."""
    missing = np.isnan(Xc).mean(axis=0)
    cols = [n for j, n in enumerate(names) if missing[j] <= MAX_MISSING]
    p = {
        "cols": cols,
        "indicators": [n for n in MISSING_INDICATORS if n in cols],
        "onehot": {},
        "clip": {},
        "median": {},
    }
    for n in cols:
        x = Xc[:, names.index(n)]
        x = x[~np.isnan(x)]
        levels = np.unique(x)
        if 3 <= len(levels) <= MAX_LEVELS:
            p["onehot"][n] = levels[1:]  # first level dropped (bias column)
        else:
            if len(levels) >= CONTINUOUS_LEVELS:
                p["clip"][n] = float(np.percentile(x, CLIP_PCT))
                x = np.minimum(x, p["clip"][n])
            p["median"][n] = float(np.median(x))
    Z, cols = _build(Xc, names, p)
    p["columns"] = ["bias"] + cols
    p["mean"], p["std"] = Z.mean(axis=0), Z.std(axis=0)
    p["std"][p["std"] == 0] = 1.0  # safety: constant column
    return p


def transform(Xc, names, p):
    """Apply the learned statistics, standardize, add the bias column."""
    Z, _ = _build(Xc, names, p)
    Z = (Z - p["mean"]) / p["std"]
    return np.column_stack([np.ones(len(Z)), Z])


def prepare(x_train, x_test, names):
    """clean + fit on train + transform both. Returns (tX_train, tX_test, p)."""
    Xc_tr, kept = clean(x_train, names)
    Xc_te, _ = clean(x_test, names)
    p = fit(Xc_tr, kept)
    return transform(Xc_tr, kept, p), transform(Xc_te, kept, p), p


# ---------------- pairwise interactions (final model) ----------------


def fit_pairs(tX_train, w, K=15):
    """Pairwise products (squares included) of the K columns with the largest |w|,
    scaled with TRAINING statistics. K=15 -> 120 new columns."""
    top = np.argsort(-np.abs(w[1:]))[:K] + 1  # skip the bias
    pairs = [(a, b) for k, a in enumerate(top) for b in top[k:]]
    P = np.column_stack([tX_train[:, a] * tX_train[:, b] for a, b in pairs])
    mu, sd = P.mean(axis=0), P.std(axis=0)
    sd[sd == 0] = 1.0
    return {"pairs": pairs, "mean": mu, "std": sd}


def add_pairs(tX, pp):
    """Append the standardized pairwise products learned by fit_pairs."""
    P = np.column_stack([tX[:, a] * tX[:, b] for a, b in pp["pairs"]])
    return np.column_stack([tX, (P - pp["mean"]) / pp["std"]])
