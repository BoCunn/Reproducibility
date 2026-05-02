# Titanic Survival Prediction — Reproducibility Experiment

Runs a Random Forest classifier on the Titanic dataset 5 times with different random seeds and reports mean ± std to expose model variability.

## Requirements

```
pandas, numpy, scikit-learn
```

## Usage

Place `train.csv` in the same directory, update `TRAIN_PATH` in the script if needed, then run:

```bash
python titanic_experiments.py
```

Results are saved to `results.csv`.

## What it does

1. Prepares features (age/fare imputation, sex and embarked encoding)
2. Runs 5-fold stratified cross-validation for each of 5 random seeds
3. Reports accuracy, ROC-AUC, and F1 per seed, then mean ± std across all runs
4. Flags instability if accuracy std exceeds 0.005 and explains the cause

## Instability note

Variance across seeds comes from RandomForest's bootstrap sampling and random feature splits. Seeds are fixed globally (`random`, `numpy`) and per-model (`random_state`) to make each run reproducible individually. Increase `n_estimators` to reduce run-to-run variance.