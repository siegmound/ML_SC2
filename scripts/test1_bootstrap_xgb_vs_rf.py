import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

XGB_PRED_PATH = "xgb_clean_v3_1_fixed_test_predictions.csv"
RF_PRED_PATH = "rf_test_clean_v3_1_test_predictions.csv"
OUTPUT_PREFIX = "test1_xgb_vs_rf_bootstrap"
N_BOOTSTRAPS = 2000
RANDOM_STATE = 42

def load_and_merge():
    xgb = pd.read_csv(XGB_PRED_PATH)
    rf = pd.read_csv(RF_PRED_PATH)
    merged = xgb.merge(
        rf,
        on=["replay_id", "time_sec", "y_true"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Merged prediction dataframe is empty.")
    return merged

def compute_metrics(df):
    y_true = df["y_true"].values
    xgb_prob = df["y_prob_xgb"].values
    rf_prob = df["y_prob_rf"].values
    xgb_pred = df["y_pred_xgb"].values
    rf_pred = df["y_pred_rf"].values
    return {
        "xgb_accuracy": accuracy_score(y_true, xgb_pred),
        "rf_accuracy": accuracy_score(y_true, rf_pred),
        "xgb_auc": roc_auc_score(y_true, xgb_prob),
        "rf_auc": roc_auc_score(y_true, rf_prob),
        "xgb_logloss": log_loss(y_true, xgb_prob, labels=[0, 1]),
        "rf_logloss": log_loss(y_true, rf_prob, labels=[0, 1]),
    }

def bootstrap_by_replay(df, n_bootstraps=2000, seed=42):
    rng = np.random.default_rng(seed)
    replay_ids = df["replay_id"].unique()
    replay_to_idx = {rid: np.where(df["replay_id"].values == rid)[0] for rid in replay_ids}

    rows = []
    for b in range(n_bootstraps):
        sampled = rng.choice(replay_ids, size=len(replay_ids), replace=True)
        idx_parts = [replay_to_idx[rid] for rid in sampled]
        boot_idx = np.concatenate(idx_parts)
        boot_df = df.iloc[boot_idx]
        m = compute_metrics(boot_df)
        rows.append({
            "bootstrap_id": b,
            "xgb_accuracy": m["xgb_accuracy"],
            "rf_accuracy": m["rf_accuracy"],
            "diff_accuracy_xgb_minus_rf": m["xgb_accuracy"] - m["rf_accuracy"],
            "xgb_auc": m["xgb_auc"],
            "rf_auc": m["rf_auc"],
            "diff_auc_xgb_minus_rf": m["xgb_auc"] - m["rf_auc"],
            "xgb_logloss": m["xgb_logloss"],
            "rf_logloss": m["rf_logloss"],
            "diff_logloss_rf_minus_xgb": m["rf_logloss"] - m["xgb_logloss"],
        })
    return pd.DataFrame(rows)

def ci95(series):
    return float(np.quantile(series, 0.025)), float(np.quantile(series, 0.975))

def main():
    merged = load_and_merge()
    point = compute_metrics(merged)

    print("Merged rows:", len(merged))
    print("Unique test replays:", merged["replay_id"].nunique())
    print("\nPoint estimates on merged test predictions")
    print(json.dumps(point, indent=2))

    boot = bootstrap_by_replay(merged, n_bootstraps=N_BOOTSTRAPS, seed=RANDOM_STATE)
    boot_csv = f"{OUTPUT_PREFIX}_samples.csv"
    boot.to_csv(boot_csv, index=False)

    summary = {
        "n_rows_merged": int(len(merged)),
        "n_replays_test": int(merged["replay_id"].nunique()),
        "point_estimates": point,
        "diff_accuracy_xgb_minus_rf_mean": float(boot["diff_accuracy_xgb_minus_rf"].mean()),
        "diff_accuracy_xgb_minus_rf_ci95": ci95(boot["diff_accuracy_xgb_minus_rf"]),
        "diff_auc_xgb_minus_rf_mean": float(boot["diff_auc_xgb_minus_rf"].mean()),
        "diff_auc_xgb_minus_rf_ci95": ci95(boot["diff_auc_xgb_minus_rf"]),
        "diff_logloss_rf_minus_xgb_mean": float(boot["diff_logloss_rf_minus_xgb"].mean()),
        "diff_logloss_rf_minus_xgb_ci95": ci95(boot["diff_logloss_rf_minus_xgb"]),
        "p_xgb_better_accuracy": float((boot["diff_accuracy_xgb_minus_rf"] > 0).mean()),
        "p_xgb_better_auc": float((boot["diff_auc_xgb_minus_rf"] > 0).mean()),
        "p_xgb_better_logloss": float((boot["diff_logloss_rf_minus_xgb"] > 0).mean()),
        "interpretation": {
            "accuracy": "Positive values favor XGBoost.",
            "auc": "Positive values favor XGBoost.",
            "logloss": "Positive values favor XGBoost because the quantity is RF logloss minus XGBoost logloss.",
        },
    }

    summary_json = f"{OUTPUT_PREFIX}_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nBootstrap summary")
    print(json.dumps(summary, indent=2))
    print("\nFiles saved:")
    print("-", boot_csv)
    print("-", summary_json)

if __name__ == "__main__":
    main()
