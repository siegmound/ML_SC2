from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .modeling import make_numeric_preprocessor


@dataclass
class TorchCandidate:
    scaler: str
    hidden_dims: tuple[int, ...]
    dropout: float
    activation: str
    batch_norm: bool
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int

    def to_dict(self) -> dict:
        return {
            "scaler": self.scaler,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
        }


def default_torch_candidates() -> list[TorchCandidate]:
    return [
        TorchCandidate("standard", (128, 64), 0.20, "relu", True, 1e-3, 1e-4, 256, 80, 10),
        TorchCandidate("standard", (256, 128), 0.25, "relu", True, 1e-3, 1e-4, 256, 100, 12),
        TorchCandidate("standard", (256, 128), 0.30, "leaky_relu", True, 8e-4, 1e-4, 256, 100, 12),
        TorchCandidate("quantile", (128, 64), 0.20, "leaky_relu", True, 1e-3, 1e-4, 256, 80, 10),
        TorchCandidate("quantile", (256, 128), 0.30, "leaky_relu", True, 8e-4, 5e-4, 256, 100, 12),
        TorchCandidate("quantile", (256, 128, 64), 0.35, "leaky_relu", True, 8e-4, 5e-4, 256, 120, 15),
    ]


def _build_torch_model(input_dim: int, candidate: TorchCandidate):
    import torch
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev = input_dim
    act: nn.Module
    for width in candidate.hidden_dims:
        layers.append(nn.Linear(prev, width))
        if candidate.batch_norm:
            layers.append(nn.BatchNorm1d(width))
        if candidate.activation == "leaky_relu":
            act = nn.LeakyReLU(0.1)
        else:
            act = nn.ReLU()
        layers.append(act)
        if candidate.dropout > 0:
            layers.append(nn.Dropout(candidate.dropout))
        prev = width
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def fit_torch_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    candidate: TorchCandidate,
    seed: int,
    device: str = "cpu",
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prep = make_numeric_preprocessor(candidate.scaler)
    X_train_t = prep.fit_transform(X_train)
    X_val_t = prep.transform(X_val)

    X_train_arr = np.asarray(X_train_t, dtype=np.float32)
    X_val_arr = np.asarray(X_val_t, dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    y_val_arr = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

    train_ds = TensorDataset(torch.from_numpy(X_train_arr), torch.from_numpy(y_train_arr))
    train_loader = DataLoader(train_ds, batch_size=candidate.batch_size, shuffle=True)

    resolved_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    model = _build_torch_model(X_train_arr.shape[1], candidate).to(resolved_device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate.lr, weight_decay=candidate.weight_decay)

    best_state = None
    best_val_loss = None
    best_epoch = -1
    wait = 0
    history: list[dict] = []

    x_val_tensor = torch.from_numpy(X_val_arr).to(resolved_device)
    y_val_tensor = torch.from_numpy(y_val_arr).to(resolved_device)

    for epoch in range(1, candidate.max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(resolved_device)
            yb = yb.to(resolved_device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_tensor)
            val_loss = float(criterion(val_logits, y_val_tensor).item())
            val_prob = torch.sigmoid(val_logits).detach().cpu().numpy().reshape(-1)

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)) if batch_losses else np.nan,
            "val_loss": val_loss,
        })

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= candidate.patience:
                break

    if best_state is None:
        raise RuntimeError("Torch training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_logits = model(x_val_tensor)
        val_prob = torch.sigmoid(val_logits).detach().cpu().numpy().reshape(-1)

    return {
        "preprocessor": prep,
        "model": model,
        "device": resolved_device,
        "val_prob": val_prob,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "history": history,
    }


def predict_torch_model(model_bundle: dict, X: pd.DataFrame) -> np.ndarray:
    import torch

    prep = model_bundle["preprocessor"]
    model = model_bundle["model"]
    resolved_device = model_bundle["device"]
    X_arr = np.asarray(prep.transform(X), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_arr).to(resolved_device))
        prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    return prob
