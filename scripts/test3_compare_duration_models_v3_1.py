import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------
XGB_PRED_PATH = "xgb_clean_v3_1_fixed_test_predictions.csv"
RF_PRED_PATH = "rf_test_clean_v3_1_test_predictions.csv"
MLP_DURATION_PATH = "mlp_torch_gpu_v3_1_accuracy_vs_duration.csv"

OUTPUT_PREFIX = "test3_duration_comparison_v3_1"

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def duration_table_from_predictions(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    needed = {"replay_id", "time_sec", "y_true", pred_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions file: {sorted(missing)}")

    test_df = pd.DataFrame({
        "replay_id": df["replay_id"].values,
        "correct": (df[pred_col].astype(int).values == df["y_true"].astype(int).values).astype(int),
        "snapshot_time": df["time_sec"].values,
    })

    durations = test_df.groupby("replay_id")["snapshot_time"].max().reset_index()
    durations.columns = ["replay_id", "match_length_sec"]

    analysis_df = test_df.merge(durations, on="replay_id", how="left")
    analysis_df["duration_min_bin"] = (analysis_df["match_length_sec"] // 300) * 5

    out = (
        analysis_df.groupby("duration_min_bin")
        .agg(
            accuracy=("correct", "mean"),
            n_samples=("correct", "size"),
            n_replays=("replay_id", "nunique"),
        )
        .reset_index()
        .sort_values("duration_min_bin")
    )
    return out

def load_mlp_duration_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"duration_min_bin", "accuracy", "n_samples", "n_replays"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in MLP duration file: {sorted(missing)}")
    return df.sort_values("duration_min_bin").reset_index(drop=True)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    xgb_pred = pd.read_csv(XGB_PRED_PATH)
    rf_pred = pd.read_csv(RF_PRED_PATH)
    mlp_dur = load_mlp_duration_csv(MLP_DURATION_PATH)

    xgb_dur = duration_table_from_predictions(xgb_pred, pred_col="y_pred_xgb")
    rf_dur = duration_table_from_predictions(rf_pred, pred_col="y_pred_rf")

    xgb_dur = xgb_dur.rename(columns={
        "accuracy": "xgb_accuracy",
        "n_samples": "xgb_n_samples",
        "n_replays": "xgb_n_replays",
    })
    rf_dur = rf_dur.rename(columns={
        "accuracy": "rf_accuracy",
        "n_samples": "rf_n_samples",
        "n_replays": "rf_n_replays",
    })
    mlp_dur = mlp_dur.rename(columns={
        "accuracy": "mlp_accuracy",
        "n_samples": "mlp_n_samples",
        "n_replays": "mlp_n_replays",
    })

    merged = xgb_dur.merge(rf_dur, on="duration_min_bin", how="outer")
    merged = merged.merge(mlp_dur, on="duration_min_bin", how="outer")
    merged = merged.sort_values("duration_min_bin").reset_index(drop=True)

    # Sample/replay consistency check across models
    merged["n_samples_consistent"] = (
        merged["xgb_n_samples"].fillna(-1).astype(int)
        == merged["rf_n_samples"].fillna(-1).astype(int)
    ) & (
        merged["xgb_n_samples"].fillna(-1).astype(int)
        == merged["mlp_n_samples"].fillna(-1).astype(int)
    )
    merged["n_replays_consistent"] = (
        merged["xgb_n_replays"].fillna(-1).astype(int)
        == merged["rf_n_replays"].fillna(-1).astype(int)
    ) & (
        merged["xgb_n_replays"].fillna(-1).astype(int)
        == merged["mlp_n_replays"].fillna(-1).astype(int)
    )

    out_csv = f"{OUTPUT_PREFIX}.csv"
    merged.to_csv(out_csv, index=False)

    # Main comparison plot
    plt.figure(figsize=(9, 6))
    plt.plot(merged["duration_min_bin"], merged["xgb_accuracy"], marker="o", label="XGBoost")
    plt.plot(merged["duration_min_bin"], merged["rf_accuracy"], marker="o", label="Random Forest")
    plt.plot(merged["duration_min_bin"], merged["mlp_accuracy"], marker="o", label="MLP PyTorch")
    plt.axhline(y=0.5, color="red", linestyle="--", label="Random guess (50%)")
    plt.xlabel("Match duration bin (minutes)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs match duration on v3_1_fixed")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = f"{OUTPUT_PREFIX}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Auxiliary sample-count plot
    plt.figure(figsize=(9, 5))
    plt.bar(merged["duration_min_bin"].astype(str), merged["xgb_n_samples"])
    plt.xlabel("Match duration bin (minutes)")
    plt.ylabel("Number of test snapshots")
    plt.title("Test sample count by duration bin")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    sample_plot_path = f"{OUTPUT_PREFIX}_sample_counts.png"
    plt.savefig(sample_plot_path, dpi=150)
    plt.close()

    summary = {
        "dataset": "v3_1_fixed",
        "models": ["XGBoost", "Random Forest", "MLP PyTorch"],
        "duration_bins": merged["duration_min_bin"].tolist(),
        "xgb_accuracy": merged["xgb_accuracy"].tolist(),
        "rf_accuracy": merged["rf_accuracy"].tolist(),
        "mlp_accuracy": merged["mlp_accuracy"].tolist(),
        "n_samples_consistent_all_bins": bool(merged["n_samples_consistent"].all()),
        "n_replays_consistent_all_bins": bool(merged["n_replays_consistent"].all()),
        "notes": "XGBoost and RF durations are recomputed from test prediction CSVs. MLP durations are loaded from the saved duration CSV generated by the corrected v3 script."
    }
    summary_path = f"{OUTPUT_PREFIX}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Merged duration comparison:")
    print(merged.to_string(index=False))
    print("\nFiles saved:")
    print("-", out_csv)
    print("-", plot_path)
    print("-", sample_plot_path)
    print("-", summary_path)

if __name__ == "__main__":
    main()
