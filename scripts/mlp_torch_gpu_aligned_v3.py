import json
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42
TARGET = "p1_wins"
GROUP = "replay_id"
TIME_COL = "time_sec"

DATASET_PATH = "starcraft_full_dataset_v3_1_fixed.csv"
OUTPUT_PREFIX = "mlp_torch_gpu_v3_1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()

CANDIDATES = [
    {"hidden_dims": [256, 128, 64], "dropout": 0.10, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 4096},
    {"hidden_dims": [512, 256, 128], "dropout": 0.15, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 4096},
    {"hidden_dims": [256, 128], "dropout": 0.10, "lr": 5e-4, "weight_decay": 1e-4, "batch_size": 2048},
    {"hidden_dims": [512, 256], "dropout": 0.20, "lr": 5e-4, "weight_decay": 1e-3, "batch_size": 4096},
]

MAX_EPOCHS = 40
PATIENCE = 6
NUM_WORKERS = 0
POS_WEIGHT_ENABLED = True

def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    groups_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    groups_test: pd.Series
    time_test: pd.Series | None

class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    drop_cols = [TARGET, GROUP]
    if TIME_COL in df.columns:
        drop_cols.append(TIME_COL)

    X = df.drop(columns=drop_cols).astype(np.float32)
    y = df[TARGET].astype(np.float32)
    groups = df[GROUP]
    times = df[TIME_COL] if TIME_COL in df.columns else None
    return df, X, y, groups, times

def split_train_val_test(X, y, groups, times=None):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss1.split(X, y, groups=groups))

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    groups_test = groups.iloc[test_idx]
    time_test = times.iloc[test_idx] if times is not None else None

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss2.split(X_train_val, y_train_val, groups=groups_train_val))

    return SplitData(
        X_train=X_train_val.iloc[train_idx],
        y_train=y_train_val.iloc[train_idx],
        groups_train=groups_train_val.iloc[train_idx],
        X_val=X_train_val.iloc[val_idx],
        y_val=y_train_val.iloc[val_idx],
        groups_val=groups_train_val.iloc[val_idx],
        X_test=X_test,
        y_test=y_test,
        groups_test=groups_test,
        time_test=time_test,
    )

def make_loaders(split: SplitData, batch_size: int):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(split.X_train).astype(np.float32)
    X_val = scaler.transform(split.X_val).astype(np.float32)
    X_test = scaler.transform(split.X_test).astype(np.float32)

    y_train = split.y_train.values.astype(np.float32)
    y_val = split.y_val.values.astype(np.float32)
    y_test = split.y_test.values.astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    return scaler, train_loader, val_loader, test_loader, X_train.shape[1]

@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
        ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)

def train_one_candidate(split: SplitData, config: dict, input_dim: int, train_loader, val_loader):
    model = TabularMLP(input_dim=input_dim, hidden_dims=config["hidden_dims"], dropout=config["dropout"]).to(DEVICE)

    if POS_WEIGHT_ENABLED:
        pos = float(split.y_train.sum())
        neg = float(len(split.y_train) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=DEVICE, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler_amp = torch.amp.GradScaler("cuda") if USE_AMP else None

    best_state = None
    best_val_logloss = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            bs = xb.shape[0]
            train_loss_sum += float(loss.detach().cpu()) * bs
            n_train += bs

        val_probs, val_true = predict_proba(model, val_loader, DEVICE)
        val_pred = (val_probs >= 0.5).astype(np.int32)
        val_logloss = log_loss(val_true, val_probs, labels=[0, 1])
        val_acc = accuracy_score(val_true, val_pred)
        val_auc = roc_auc_score(val_true, val_probs)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(n_train, 1),
            "val_logloss": val_logloss,
            "val_accuracy": val_acc,
            "val_auc": val_auc,
        })

        improved = val_logloss < (best_val_logloss - 1e-4)
        if improved:
            best_val_logloss = val_logloss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        print(
            f"[candidate={config['hidden_dims']}, epoch={epoch:02d}] "
            f"train_loss={history[-1]['train_loss']:.4f} "
            f"val_logloss={val_logloss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_auc={val_auc:.4f}"
        )

        if epochs_without_improve >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "config": config,
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
        "history": history,
        "model": model,
    }

def evaluate_test(model, test_loader):
    probs, y_true = predict_proba(model, test_loader, DEVICE)
    y_pred = (probs >= 0.5).astype(np.int32)

    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_auc": float(roc_auc_score(y_true, probs)),
        "test_logloss": float(log_loss(y_true, probs, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }
    return metrics, probs, y_pred, y_true

def save_accuracy_vs_duration(split: SplitData, y_pred):
    if split.time_test is None:
        return None

    test_df = pd.DataFrame({
        "replay_id": split.groups_test.values,
        "correct": (y_pred.astype(int) == split.y_test.values.astype(int)).astype(int),
        "snapshot_time": split.time_test.values,
    })

    durations = test_df.groupby("replay_id")["snapshot_time"].max().reset_index()
    durations.columns = ["replay_id", "match_length_sec"]

    analysis_df = test_df.merge(durations, on="replay_id", how="left")
    analysis_df["duration_min_bin"] = (analysis_df["match_length_sec"] // 300) * 5

    summary = (
        analysis_df.groupby("duration_min_bin")
        .agg(
            accuracy=("correct", "mean"),
            n_samples=("correct", "size"),
            n_replays=("replay_id", "nunique"),
        )
        .reset_index()
    )

    csv_path = f"{OUTPUT_PREFIX}_accuracy_vs_duration.csv"
    png_path = f"{OUTPUT_PREFIX}_accuracy_vs_duration.png"
    summary.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(summary["duration_min_bin"], summary["accuracy"], marker="o")
    plt.axhline(y=0.5, color="red", linestyle="--", label="Random Guess (50%)")
    plt.xlabel("Durata totale partita (minuti)")
    plt.ylabel("Accuracy")
    plt.title("PyTorch MLP Accuracy vs Match Duration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    return summary

def save_feature_importance_proxy(model, feature_names):
    first_linear = None
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            first_linear = layer
            break
    if first_linear is None:
        return None

    weights = first_linear.weight.detach().cpu().numpy()
    proxy = np.mean(np.abs(weights), axis=0)

    imp_df = pd.DataFrame({"feature": feature_names, "importance_proxy": proxy}).sort_values("importance_proxy", ascending=False)
    csv_path = f"{OUTPUT_PREFIX}_feature_importance_proxy.csv"
    imp_df.to_csv(csv_path, index=False)
    return imp_df

def main():
    set_seed()
    print(f"Device: {DEVICE} | AMP: {USE_AMP}")
    print("Loading dataset...")
    df, X, y, groups, times = load_dataset(DATASET_PATH)
    split = split_train_val_test(X, y, groups, times=times)

    best_run = None
    best_test_loader = None

    for idx, candidate in enumerate(CANDIDATES, start=1):
        print("\n" + "=" * 80)
        print(f"CANDIDATE {idx}/{len(CANDIDATES)} -> {candidate}")
        print("=" * 80)

        _, train_loader, val_loader, test_loader, input_dim = make_loaders(split, candidate["batch_size"])
        run = train_one_candidate(split, candidate, input_dim, train_loader, val_loader)

        if best_run is None or run["best_val_logloss"] < best_run["best_val_logloss"]:
            best_run = run
            best_test_loader = test_loader

    print("\n" + "=" * 80)
    print("BEST CONFIG")
    print("=" * 80)
    print(best_run["config"])
    print(f"Best epoch      : {best_run['best_epoch']}")
    print(f"Best val logloss: {best_run['best_val_logloss']:.6f}")

    metrics, probs, y_pred, y_true = evaluate_test(best_run["model"], best_test_loader)

    print("\n" + "=" * 40)
    print("VALUTAZIONE FINALE SU TEST PURO")
    print("=" * 40)
    print(f"Accuracy          : {metrics['test_accuracy']:.4f}")
    print(f"Balanced Accuracy : {metrics['test_balanced_accuracy']:.4f}")
    print(f"ROC-AUC           : {metrics['test_auc']:.4f}")
    print(f"Log Loss          : {metrics['test_logloss']:.4f}\n")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification Report:")
    print(metrics["classification_report"])

    imp_df = save_feature_importance_proxy(best_run["model"], split.X_train.columns)
    if imp_df is not None:
        print("\n--- TOP FEATURE IMPORTANCE PROXY ---")
        print(imp_df.head(15).to_string(index=False))

    duration_summary = save_accuracy_vs_duration(split, y_pred)
    if duration_summary is not None:
        print("\n--- ACCURACY VS DURATA (con sample count) ---")
        print(duration_summary.to_string(index=False))

    summary = {
        "model": "MLP_PyTorch",
        "dataset_path": DATASET_PATH,
        "device": DEVICE,
        "amp": USE_AMP,
        "rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "n_replays_total": int(groups.nunique()),
        "n_replays_train": int(split.groups_train.nunique()),
        "n_replays_val": int(split.groups_val.nunique()),
        "n_replays_test": int(split.groups_test.nunique()),
        "best_params": best_run["config"],
        "best_epoch": int(best_run["best_epoch"]),
        "cv_accuracy": None,
        "selection_metric": "validation_logloss",
        "validation_logloss": float(best_run["best_val_logloss"]),
        "test_accuracy": float(metrics["test_accuracy"]),
        "test_balanced_accuracy": float(metrics["test_balanced_accuracy"]),
        "test_auc": float(metrics["test_auc"]),
        "test_logloss": float(metrics["test_logloss"]),
    }

    with open(f"{OUTPUT_PREFIX}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nAnalisi completata.")
    print("File salvati:")
    print(f"- {OUTPUT_PREFIX}_feature_importance_proxy.csv")
    print(f"- {OUTPUT_PREFIX}_accuracy_vs_duration.csv")
    print(f"- {OUTPUT_PREFIX}_accuracy_vs_duration.png")
    print(f"- {OUTPUT_PREFIX}_summary.json")

if __name__ == "__main__":
    main()
