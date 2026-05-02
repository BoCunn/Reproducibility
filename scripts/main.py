"""
Titanic — Reproducibility Experiment
Fixed seeds, 5 runs, mean ± std reported.
"""

import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH = "data/train.csv"
SEEDS      = [42, 7, 123, 999, 2024]
N_SPLITS   = 5

# ── Load & prepare data ───────────────────────────────────────────────────────
df = pd.read_csv(TRAIN_PATH)

df["Sex"]      = (df["Sex"] == "male").astype(int)
df["Age"]      = df["Age"].fillna(df["Age"].median())
df["Fare"]     = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna("S").map({"S": 0, "C": 1, "Q": 2})

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X = df[FEATURES].values
y = df["Survived"].values

# ── Experiment loop ───────────────────────────────────────────────────────────
results = []

for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)

    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    cv    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    acc_scores, auc_scores, f1_scores = [], [], []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        preds  = model.predict(X_val)
        probas = model.predict_proba(X_val)[:, 1]

        acc_scores.append(accuracy_score(y_val, preds))
        auc_scores.append(roc_auc_score(y_val, probas))
        f1_scores.append(f1_score(y_val, preds))

    results.append({
        "seed":     seed,
        "accuracy": np.mean(acc_scores),
        "roc_auc":  np.mean(auc_scores),
        "f1":       np.mean(f1_scores),
    })
    print(f"Seed {seed:>4}: acc={results[-1]['accuracy']:.4f}  "
          f"auc={results[-1]['roc_auc']:.4f}  f1={results[-1]['f1']:.4f}")

# ── Report ────────────────────────────────────────────────────────────────────
acc = [r["accuracy"] for r in results]
auc = [r["roc_auc"]  for r in results]
f1  = [r["f1"]       for r in results]

print("\n── Results (RandomForest, 5 runs x 5-fold CV) ──────────────────────────────")
print(f"Accuracy : {np.mean(acc):.4f} +/- {np.std(acc):.4f}")
print(f"ROC-AUC  : {np.mean(auc):.4f} +/- {np.std(auc):.4f}")
print(f"F1       : {np.mean(f1):.4f}  +/- {np.std(f1):.4f}")

# Instability note
if np.std(acc) > 0.005:
    print("\nInstability detected (Accuracy std > 0.005).")
    print("Cause: RandomForest uses bootstrap sampling and random feature splits,")
    print("so different seeds produce slightly different ensembles.")
    print("Mitigation: seeds are fixed per run; raise n_estimators to reduce variance.")

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame(results).to_csv("data/results.csv", index=False)
print("\nResults saved to results.csv")