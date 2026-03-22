import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit

from project_paths import artifact_path, dataset_path

RANDOM_STATE = 42
TARGET = "p1_wins"
GROUP = "replay_id"
TIME = "time_sec"
DATASET_PATH = dataset_path("starcraft_full_dataset_v3_1_fixed.csv")
OUTPUT_PREFIX = "rf_test_clean_v3_1"

print("Caricamento dataset...")
df = pd.read_csv(DATASET_PATH)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop(columns=[TARGET, GROUP, TIME]).astype(np.float32)
y = df[TARGET].astype(np.int32)
groups = df[GROUP]
times = df[TIME]

gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_val_idx, test_idx = next(gss_outer.split(X, y, groups=groups))

X_train_val = X.iloc[train_val_idx]
y_train_val = y.iloc[train_val_idx]
groups_train_val = groups.iloc[train_val_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]
groups_test = groups.iloc[test_idx]
times_test = times.iloc[test_idx]

gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx_rel, val_idx_rel = next(
    gss_inner.split(X_train_val, y_train_val, groups=groups_train_val)
)

X_train = X_train_val.iloc[train_idx_rel]
y_train = y_train_val.iloc[train_idx_rel]
groups_train = groups_train_val.iloc[train_idx_rel]

X_val = X_train_val.iloc[val_idx_rel]
y_val = y_train_val.iloc[val_idx_rel]
groups_val = groups_train_val.iloc[val_idx_rel]

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [None, 20],
    "min_samples_split": [10],
    "min_samples_leaf": [4],
    "max_features": ["sqrt"],
}

cv_strategy = GroupKFold(n_splits=3)
grid_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    param_grid=param_grid,
    cv=cv_strategy,
    scoring="accuracy",
    verbose=2,
    n_jobs=1,
)
grid_rf.fit(X_train, y_train, groups=groups_train)

print(f"\nMigliori parametri RF: {grid_rf.best_params_}")
print(f"Miglior score CV (train only): {grid_rf.best_score_:.4f}")

best_rf = grid_rf.best_estimator_
print("\n--- FASE 2: Fit finale su TRAIN, validation usata solo per check ---")
best_rf.fit(X_train, y_train)

y_prob = best_rf.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(np.int32)

acc = accuracy_score(y_test, y_pred)
bacc = balanced_accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
ll = log_loss(y_test, y_prob)

print("\n" + "=" * 40)
print("VALUTAZIONE FINALE SU TEST PURO")
print("=" * 40)
print(f"Accuracy : {acc:.4f}")
print(f"Balanced Accuracy : {bacc:.4f}")
print(f"ROC-AUC : {auc:.4f}")
print(f"Log Loss : {ll:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

pred_df = pd.DataFrame(
    {
        "replay_id": groups_test.values,
        "time_sec": times_test.values,
        "y_true": y_test.values,
        "y_prob_rf": y_prob,
        "y_pred_rf": y_pred,
    }
)
pred_csv = artifact_path(f"{OUTPUT_PREFIX}_test_predictions.csv")
pred_df.to_csv(pred_csv, index=False)

summary = {
    "model": "random_forest_export_preserved",
    "artifact_line": "export_preserved_test1_to_test4",
    "script": "scripts/test1_rf_export_predictions_v3_1.py",
    "dataset_path": "datasets/starcraft_full_dataset_v3_1_fixed.csv",
    "rows": int(len(df)),
    "n_features": int(X.shape[1]),
    "n_replays_total": int(groups.nunique()),
    "n_replays_train": int(groups_train.nunique()),
    "n_replays_val": int(groups_val.nunique()),
    "n_replays_test": int(groups_test.nunique()),
    "best_params": grid_rf.best_params_,
    "cv_accuracy": float(grid_rf.best_score_),
    "test_accuracy": float(acc),
    "test_balanced_accuracy": float(bacc),
    "test_auc": float(auc),
    "test_logloss": float(ll),
}
summary_json = artifact_path(f"{OUTPUT_PREFIX}_summary.json")
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nFile salvati:")
print(f"- {pred_csv}")
print(f"- {summary_json}")
