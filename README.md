# Heart disease prediction (CS-433 Project 1)

Regularized logistic regression with pairwise interactions, written with NumPy only.
Cross-validated F1: 0.429. AIcrowd F1: 0.442.

## Run

Put `x_train.csv`, `y_train.csv` and `x_test.csv` in `dataset/`, then from the repository root:

```
python run.py           # writes submission.csv (our final submission)
python make_figures.py  # makes the report figures in results/
```

## Files

- `implementations.py`: the 6 required methods
- `preprocessing.py`: data cleaning and feature processing
- `cv.py`: F1 score, threshold, 5-fold cross-validation
- `run.py`: trains the final model and writes the submission
- `make_figures.py`: figures of the report
- `helpers.py`: loading and submission functions from the course

## Data preparation

- Drop columns that are not about the person (dates, ids, weights), redundant versions of the
  same variable, and `HAREHAB1` (only answered by heart patients, it reveals the label).
- Survey codes for "don't know" / "refused" (7, 9, 77, 99, ...) become missing; 88 ("none") becomes 0.
- `BPMEDS` and `POORHLTH` are only asked when relevant: missing means "no" and 0.
- On the training data only: drop columns with more than 50% missing, one-hot encode columns with
  3 to 10 values, clip continuous columns at the 99th percentile, add 3 missing indicators,
  fill missing values with the median, standardize, add a bias (131 columns).
- Final model only: add the products of all pairs of the 15 columns with the largest weights
  in a first model (120 columns).

## Model and validation

- Regularized logistic regression: λ = 1e-4, 500 gradient descent steps, γ = 0.5.
- Threshold chosen to maximize F1 on the training data (only 8.8% of people are positive).
- 5-fold cross-validation; preprocessing, interactions and threshold are fitted on the training folds.

## Team

- Erik Maffei
- Miray Tümer
- Karl Uriot
