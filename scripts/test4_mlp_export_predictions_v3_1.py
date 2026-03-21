import json
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from project_paths import artifact_path, dataset_path

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

DATASET_PATH = dataset_path("starcraft_full_dataset_v3_1_fixed.csv")
OUTPUT_PREFIX = "mlp_torch_gpu_v3_1_export"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()

# Best config from previous MLP calibration run
BEST_CONFIG = {
    "hidden_dims": [512, 256],
    "dropout": 0.2,
    "lr": 5e-4,
    "weight_decay": 1e-3,
    "batch_size": 4096,
}
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
def predict_proba(model, loader):
    model.eval()
    probs = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
        ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)

def train_best_model(split: SplitData):
    _, train_loader, val_loader, test_loader, input_dim = make_loaders(split, BEST_CONFIG["batch_size"])

    model = TabularMLP(input_dim=input_dim, hidden_dims=BEST_CONFIG["hidden_dims"], dropout=BEST_CONFIG["dropout"]).to(DEVICE)

    if POS_WEIGHT_ENABLED:
        pos = float(split.y_train.sum())
        neg = float(len(split.y_train) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=DEVICE, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=BEST_CONFIG["lr"], weight_decay=BEST_CONFIG["weight_decay"])
    scaler_amp = torch.amp.GradScaler("cuda") if USE_AMP else None

    best_state = None
    best_val_logloss = float("inf")
    best_epoch = -1
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
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

        val_prob, val_true = predict_proba(model, val_loader)
        val_ll = log_loss(val_true, val_prob, labels=[0, 1])
        print(f"[epoch={epoch:02d}] val_logloss={val_ll:.6f}")

        if val_ll < best_val_logloss - 1e-4:
            best_val_logloss = val_ll
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, test_loader, best_epoch, best_val_logloss

def main():
    set_seed()
    print(f"Device: {DEVICE} | AMP: {USE_AMP}")
    df, X, y, groups, times = load_dataset(DATASET_PATH)
    split = split_train_val_test(X, y, groups, times=times)

    model, test_loader, best_epoch, best_val_logloss = train_best_model(split)
    y_prob, y_true = predict_proba(model, test_loader)
    y_pred = (y_prob >= 0.5).astype(np.int32)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob, labels=[0, 1])

    print("\n" + "=" * 40)
    print("VALUTAZIONE FINALE SU TEST PURO")
    print("=" * 40)
    print(f"Accuracy          : {acc:.4f}")
    print(f"Balanced Accuracy : {bacc:.4f}")
    print(f"ROC-AUC           : {auc:.4f}")
    print(f"Log Loss          : {ll:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    pred_df = pd.DataFrame({
        "replay_id": split.groups_test.values,
        "time_sec": split.time_test.values,
        "y_true": split.y_test.values.astype(int),
        "y_prob_mlp": y_prob,
        "y_pred_mlp": y_pred,
    })
    pred_csv = artifact_path(f"{OUTPUT_PREFIX}_test_predictions.csv")
    pred_df.to_csv(pred_csv, index=False)

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
        "best_params": BEST_CONFIG,
        "best_epoch": int(best_epoch),
        "validation_logloss": float(best_val_logloss),
        "test_accuracy": float(acc),
        "test_balanced_accuracy": float(bacc),
        "test_auc": float(auc),
        "test_logloss": float(ll),
    }
    summary_json = artifact_path(f"{OUTPUT_PREFIX}_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFile salvati:")
    print("-", pred_csv)
    print("-", summary_json)

if __name__ == "__main__":
    main()
