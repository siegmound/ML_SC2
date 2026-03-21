import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from project_paths import artifact_path, figure_path

XGB_PRED_PATH = artifact_path("xgb_clean_v3_1_fixed_test_predictions.csv")
RF_PRED_PATH = artifact_path("rf_test_clean_v3_1_test_predictions.csv")
MLP_PRED_PATH = artifact_path("mlp_torch_gpu_v3_1_export_test_predictions.csv")
N_BINS = 10

def calibration_table(y_true, y_prob, n_bins=10):
    df = pd.DataFrame({
        "y_true": np.asarray(y_true, dtype=np.int32),
        "y_prob": np.asarray(y_prob, dtype=np.float64),
    })
    df["bin"] = pd.cut(
        df["y_prob"],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
        duplicates="drop",
    )
    table = (
        df.groupby("bin", observed=False)
        .agg(
            n_samples=("y_true", "size"),
            mean_pred=("y_prob", "mean"),
            frac_positive=("y_true", "mean"),
        )
        .reset_index()
    )
    total = max(len(df), 1)
    table["abs_gap"] = (table["mean_pred"] - table["frac_positive"]).abs()
    table["weight"] = table["n_samples"] / total
    ece = float((table["abs_gap"] * table["weight"]).sum())
    mce = float(table["abs_gap"].max()) if len(table) else float("nan")
    return table, ece, mce

def summarize_model(label, y_true, y_prob):
    table, ece, mce = calibration_table(y_true, y_prob, n_bins=N_BINS)
    summary = {
        "model": label,
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "mean_predicted_probability": float(np.mean(y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "ece": float(ece),
        "mce": float(mce),
    }
    return table, summary

def main():
    xgb = pd.read_csv(XGB_PRED_PATH)
    rf = pd.read_csv(RF_PRED_PATH)
    mlp = pd.read_csv(MLP_PRED_PATH)

    merged = xgb.merge(rf, on=["replay_id", "time_sec", "y_true"], how="inner", validate="one_to_one")
    merged = merged.merge(mlp, on=["replay_id", "time_sec", "y_true"], how="inner", validate="one_to_one")

    y_true = merged["y_true"].values
    xgb_table, xgb_summary = summarize_model("XGBoost", y_true, merged["y_prob_xgb"].values)
    rf_table, rf_summary = summarize_model("Random Forest", y_true, merged["y_prob_rf"].values)
    mlp_table, mlp_summary = summarize_model("MLP PyTorch", y_true, merged["y_prob_mlp"].values)

    xgb_table.to_csv(artifact_path("test4_xgb_calibration_bins.csv"), index=False)
    rf_table.to_csv(artifact_path("test4_rf_calibration_bins.csv"), index=False)
    mlp_table.to_csv(artifact_path("test4_mlp_calibration_bins.csv"), index=False)

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    for label, table in [("XGBoost", xgb_table), ("Random Forest", rf_table), ("MLP PyTorch", mlp_table)]:
        plot_df = table.dropna(subset=["mean_pred", "frac_positive"])
        plt.plot(plot_df["mean_pred"], plot_df["frac_positive"], marker="o", label=label)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive frequency")
    plt.title("Reliability diagram on v3_1_fixed test set")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path("test4_reliability_xgb_rf_mlp.png"), dpi=150)
    plt.close()

    summary = {
        "dataset": "v3_1_fixed",
        "n_samples": int(len(y_true)),
        "xgboost": xgb_summary,
        "random_forest": rf_summary,
        "mlp_pytorch": mlp_summary,
    }
    with open(artifact_path("test4_calibration_summary_xgb_rf_mlp.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Calibration summary:")
    print(json.dumps(summary, indent=2))
    print("\nFiles saved:")
    print("- test4_xgb_calibration_bins.csv")
    print("- test4_rf_calibration_bins.csv")
    print("- test4_mlp_calibration_bins.csv")
    print("- test4_reliability_xgb_rf_mlp.png")
    print("- test4_calibration_summary_xgb_rf_mlp.json")

if __name__ == "__main__":
    main()
