import json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss,
)

RANDOM_STATE = 42
TARGET = "p1_wins"
GROUP = "replay_id"
TIME = "time_sec"

DATASET_PATH = "starcraft_full_dataset_v3_1_fixed.csv"
OUTPUT_PREFIX = "xgb_clean_v3_1_fixed"

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

xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    device="cuda",
    random_state=RANDOM_STATE,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
)

param_grid = {
    "max_depth": [4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05],
}

cv_strategy = GroupKFold(n_splits=3)
grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring="accuracy",
    verbose=2,
    n_jobs=1,
)
grid.fit(X_train, y_train, groups=groups_train)

print(f"\nMigliori parametri strutturali: {grid.best_params_}")
print(f"Miglior score CV (train only): {grid.best_score_:.4f}")

final_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    device="cuda",
    random_state=RANDOM_STATE,
    max_depth=grid.best_params_["max_depth"],
    learning_rate=grid.best_params_["learning_rate"],
    n_estimators=2000,
    early_stopping_rounds=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
)
final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

best_iteration = int(getattr(final_model, "best_iteration", final_model.n_estimators))
y_prob = final_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(np.int32)

acc = accuracy_score(y_test, y_pred)
bacc = balanced_accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
ll = log_loss(y_test, y_prob)

print("\n" + "=" * 40)
print("VALUTAZIONE FINALE SU TEST PURO")
print("=" * 40)
print(f"Accuracy          : {acc:.4f}")
print(f"Balanced Accuracy : {bacc:.4f}")
print(f"ROC-AUC           : {auc:.4f}")
print(f"Log Loss          : {ll:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

pred_df = pd.DataFrame({
    "replay_id": groups_test.values,
    "time_sec": times_test.values,
    "y_true": y_test.values,
    "y_prob_xgb": y_prob,
    "y_pred_xgb": y_pred,
})
pred_csv = f"{OUTPUT_PREFIX}_test_predictions.csv"
pred_df.to_csv(pred_csv, index=False)

summary = {
    "model": "XGBoost",
    "dataset_path": DATASET_PATH,
    "rows": int(len(df)),
    "n_features": int(X.shape[1]),
    "n_replays_total": int(groups.nunique()),
    "n_replays_train": int(groups_train.nunique()),
    "n_replays_val": int(groups_val.nunique()),
    "n_replays_test": int(groups_test.nunique()),
    "best_params": grid.best_params_,
    "cv_accuracy": float(grid.best_score_),
    "best_iteration": best_iteration,
    "test_accuracy": float(acc),
    "test_balanced_accuracy": float(bacc),
    "test_auc": float(auc),
    "test_logloss": float(ll),
}
summary_json = f"{OUTPUT_PREFIX}_summary.json"
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nFile salvati:")
print(f"- {pred_csv}")
print(f"- {summary_json}")
