import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

XGB_PRED_PATH = "xgb_clean_v3_1_fixed_test_predictions.csv"
RF_PRED_PATH = "rf_test_clean_v3_1_test_predictions.csv"
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

    table["bin_left"] = table["bin"].apply(lambda x: float(x.left) if pd.notna(x) else np.nan)
    table["bin_right"] = table["bin"].apply(lambda x: float(x.right) if pd.notna(x) else np.nan)

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

def save_reliability_plot(xgb_table, rf_table):
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    xgb_plot = xgb_table.dropna(subset=["mean_pred", "frac_positive"])
    rf_plot = rf_table.dropna(subset=["mean_pred", "frac_positive"])

    plt.plot(xgb_plot["mean_pred"], xgb_plot["frac_positive"], marker="o", label="XGBoost")
    plt.plot(rf_plot["mean_pred"], rf_plot["frac_positive"], marker="o", label="Random Forest")

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive frequency")
    plt.title("Reliability diagram on v3_1_fixed test set")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("test2_reliability_xgb_vs_rf.png", dpi=150)
    plt.close()

def save_histogram_plot(xgb_prob, rf_prob):
    plt.figure(figsize=(8, 5))
    plt.hist(xgb_prob, bins=20, alpha=0.6, label="XGBoost")
    plt.hist(rf_prob, bins=20, alpha=0.6, label="Random Forest")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Probability distribution on v3_1_fixed test set")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("test2_probability_histograms_xgb_vs_rf.png", dpi=150)
    plt.close()

def main():
    xgb = pd.read_csv(XGB_PRED_PATH)
    rf = pd.read_csv(RF_PRED_PATH)

    merge_check = xgb.merge(
        rf,
        on=["replay_id", "time_sec", "y_true"],
        how="inner",
        validate="one_to_one",
    )
    if len(merge_check) != len(xgb) or len(merge_check) != len(rf):
        raise ValueError("Prediction files do not align perfectly on replay_id/time_sec/y_true.")

    y_true = merge_check["y_true"].values
    xgb_prob = merge_check["y_prob_xgb"].values
    rf_prob = merge_check["y_prob_rf"].values

    xgb_table, xgb_summary = summarize_model("XGBoost", y_true, xgb_prob)
    rf_table, rf_summary = summarize_model("Random Forest", y_true, rf_prob)

    xgb_table.to_csv("test2_xgb_calibration_bins.csv", index=False)
    rf_table.to_csv("test2_rf_calibration_bins.csv", index=False)

    combined_summary = {
        "dataset": "v3_1_fixed",
        "n_samples": int(len(y_true)),
        "xgboost": xgb_summary,
        "random_forest": rf_summary,
        "delta_brier_rf_minus_xgb": float(rf_summary["brier_score"] - xgb_summary["brier_score"]),
        "delta_logloss_rf_minus_xgb": float(rf_summary["log_loss"] - xgb_summary["log_loss"]),
        "delta_ece_rf_minus_xgb": float(rf_summary["ece"] - xgb_summary["ece"]),
        "interpretation": {
            "brier": "Lower is better.",
            "log_loss": "Lower is better.",
            "ece": "Lower is better.",
            "delta_*_rf_minus_xgb": "Positive values favor XGBoost."
        }
    }

    with open("test2_calibration_summary_xgb_vs_rf.json", "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)

    save_reliability_plot(xgb_table, rf_table)
    save_histogram_plot(xgb_prob, rf_prob)

    print("Calibration summary:")
    print(json.dumps(combined_summary, indent=2))
    print("\nFiles saved:")
    print("- test2_xgb_calibration_bins.csv")
    print("- test2_rf_calibration_bins.csv")
    print("- test2_calibration_summary_xgb_vs_rf.json")
    print("- test2_reliability_xgb_vs_rf.png")
    print("- test2_probability_histograms_xgb_vs_rf.png")

if __name__ == "__main__":
    main()
