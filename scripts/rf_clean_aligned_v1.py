import json
import os

import matplotlib.pyplot as plt
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

from project_paths import artifact_path, dataset_path, figure_path

# ============================================================
# Random Forest - clean protocol aligned with XGBoost
# - Group-aware split by replay_id
# - train / validation / test separation
# - model selection only on train
# - test touched once at the end
# ============================================================

RANDOM_STATE = 42
TARGET = "p1_wins"
GROUP = "replay_id"
TIME_COL = "time_sec"
DATASET_PATH = dataset_path("starcraft_full_dataset_v3_1_fixed.csv")
OUTPUT_PREFIX = "rf_clean_v3_1"


def load_dataset(csv_path: str):
    print("Caricamento dataset...")
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    required = {TARGET, GROUP}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    drop_cols = [TARGET, GROUP]
    if TIME_COL in df.columns:
        drop_cols.append(TIME_COL)

    X = df.drop(columns=drop_cols).astype(np.float32)
    y = df[TARGET].astype(np.int32)
    groups = df[GROUP]
    return df, X, y, groups


def split_train_val_test(X, y, groups):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss1.split(X, y, groups=groups))

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    groups_test = groups.iloc[test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss2.split(X_train_val, y_train_val, groups=groups_train_val))

    X_train = X_train_val.iloc[train_idx]
    y_train = y_train_val.iloc[train_idx]
    groups_train = groups_train_val.iloc[train_idx]
    X_val = X_train_val.iloc[val_idx]
    y_val = y_train_val.iloc[val_idx]
    groups_val = groups_train_val.iloc[val_idx]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "groups_train": groups_train,
        "X_val": X_val,
        "y_val": y_val,
        "groups_val": groups_val,
        "X_test": X_test,
        "y_test": y_test,
        "groups_test": groups_test,
    }


def fit_model(parts):
    print("\n--- FASE 1: Model selection RF su TRAIN only ---")
    rf_base = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    param_grid = {
        "n_estimators": [300],
        "max_depth": [16, 24, None],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 4],
        "max_features": ["sqrt", 0.5],
    }

    cv_strategy = GroupKFold(n_splits=3)
    grid = GridSearchCV(
        rf_base,
        param_grid,
        cv=cv_strategy,
        scoring="accuracy",
        verbose=2,
        n_jobs=-1,
    )
    grid.fit(parts["X_train"], parts["y_train"], groups=parts["groups_train"])

    print(f"Migliori parametri RF: {grid.best_params_}")
    print(f"Miglior score CV (train only): {grid.best_score_:.4f}")

    print("\n--- FASE 2: Fit finale su TRAIN, validation usata solo per check ---")
    final_model = RandomForestClassifier(
        **grid.best_params_,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    final_model.fit(parts["X_train"], parts["y_train"])
    return grid, final_model


def evaluate_model(model, grid, parts, full_df, feature_names):
    print("\n" + "=" * 40)
    print("VALUTAZIONE FINALE SU TEST PURO")
    print("=" * 40)

    y_pred = model.predict(parts["X_test"])
    y_prob = model.predict_proba(parts["X_test"])[:, 1]
    metrics = {
        "test_accuracy": accuracy_score(parts["y_test"], y_pred),
        "test_balanced_accuracy": balanced_accuracy_score(parts["y_test"], y_pred),
        "test_auc": roc_auc_score(parts["y_test"], y_prob),
        "test_logloss": log_loss(parts["y_test"], y_prob),
    }

    print(f"Accuracy : {metrics['test_accuracy']:.4f}")
    print(f"Balanced Accuracy : {metrics['test_balanced_accuracy']:.4f}")
    print(f"ROC-AUC : {metrics['test_auc']:.4f}")
    print(f"Log Loss : {metrics['test_logloss']:.4f}")

    cm = confusion_matrix(parts["y_test"], y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(parts["y_test"], y_pred, digits=4))

    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False)
    print("\n--- TOP FEATURE IMPORTANCE ---")
    print(imp_df.head(15))
    imp_df.to_csv(artifact_path(f"{OUTPUT_PREFIX}_feature_importance.csv"), index=False)

    test_groups = parts["groups_test"]
    replay_lengths = full_df.groupby(GROUP)[TIME_COL].max() / 60.0
    duration_map = replay_lengths.to_dict()
    test_meta = pd.DataFrame({
        GROUP: test_groups.values,
        "y_true": parts["y_test"].values,
        "y_pred": y_pred,
    })
    test_meta["duration_min"] = test_meta[GROUP].map(duration_map)
    test_meta["duration_min_bin"] = (np.floor(test_meta["duration_min"] / 5) * 5).astype(float)

    duration_summary = (
        test_meta.groupby("duration_min_bin")
        .apply(
            lambda g: pd.Series(
                {
                    "accuracy": accuracy_score(g["y_true"], g["y_pred"]),
                    "n_samples": len(g),
                    "n_replays": g[GROUP].nunique(),
                }
            )
        )
        .reset_index()
        .sort_values("duration_min_bin")
    )

    print("\n--- ACCURACY VS DURATA (con sample count) ---")
    print(duration_summary)
    duration_summary.to_csv(artifact_path(f"{OUTPUT_PREFIX}_accuracy_vs_duration.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(duration_summary["duration_min_bin"], duration_summary["accuracy"], marker="o")
    plt.xlabel("Match duration bin (minutes)")
    plt.ylabel("Accuracy")
    plt.title("Random Forest Accuracy vs Match Duration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_path(f"{OUTPUT_PREFIX}_accuracy_vs_duration.png"), dpi=200)
    plt.close()

    summary = {
        "model": "random_forest_aligned",
        "dataset_path": DATASET_PATH,
        "rows": int(len(full_df)),
        "n_features": int(len(feature_names)),
        "n_replays_total": int(full_df[GROUP].nunique()),
        "n_replays_train": int(parts["groups_train"].nunique()),
        "n_replays_val": int(parts["groups_val"].nunique()),
        "n_replays_test": int(parts["groups_test"].nunique()),
        "best_params": grid.best_params_,
        "cv_accuracy": float(grid.best_score_),
        **metrics,
    }
    with open(artifact_path(f"{OUTPUT_PREFIX}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(DATASET_PATH)

    full_df, X, y, groups = load_dataset(DATASET_PATH)
    parts = split_train_val_test(X, y, groups)
    grid, model = fit_model(parts)
    evaluate_model(model, grid, parts, full_df, list(X.columns))

    print("\nAnalisi completata.")
    print("File salvati:")
    print(f"- {artifact_path(f'{OUTPUT_PREFIX}_feature_importance.csv')}")
    print(f"- {artifact_path(f'{OUTPUT_PREFIX}_accuracy_vs_duration.csv')}")
    print(f"- {figure_path(f'{OUTPUT_PREFIX}_accuracy_vs_duration.png')}")
    print(f"- {artifact_path(f'{OUTPUT_PREFIX}_summary.json')}")


if __name__ == "__main__":
    main()
