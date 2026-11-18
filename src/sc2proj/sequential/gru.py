"""S4: replay-safe, validation-selected GRU on S1/S2 temporal windows.

Training never evaluates the historical test partition. Historical RF/XGB/MLP
predictions are not used for fitting or checkpoint selection.
"""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .baselines import endpoint_frame
from .data import IndexedSequences, PartitionedSequences, PreparedSequences, make_loaders


@dataclass(frozen=True)
class GRUConfig:
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    patience: int = 4
    min_delta: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 0

    def __post_init__(self) -> None:
        if any(v < 1 for v in (self.hidden_size, self.num_layers, self.batch_size,
                                self.max_epochs, self.patience)):
            raise ValueError("Hidden size, layers, batch size, epochs and patience must be positive")
        if self.num_workers < 0 or self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer or loader configuration")
        if not (0 <= self.dropout < 1) or self.min_delta < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid dropout, minimum improvement or gradient clipping")


class GRUWinner(nn.Module):
    """Unidirectional recurrent classifier; output is a single binary logit."""

    def __init__(self, feature_count: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        if feature_count < 1 or hidden_size < 1 or num_layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid GRU dimensions or dropout")
        self.feature_count = feature_count
        self.gru = nn.GRU(feature_count, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout),
                                  nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] < 1 or x.shape[2] != self.feature_count:
            raise ValueError("Expected nonempty [batch, time, feature_count] input")
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1, :]).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _epoch(model: GRUWinner, loader: DataLoader, device: torch.device,
           optimizer: torch.optim.Optimizer | None, grad_clip: float,
           max_batches: int | None = None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss, total_rows = 0.0, 0
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, (features, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, labels)
            if training:
                (loss / len(labels)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_rows += len(labels)
            total_loss += float(loss.detach().item())
    if total_rows == 0:
        raise ValueError("Empty epoch: no windows available")
    return total_loss / total_rows


def fit_gru(prepared: PreparedSequences, config: GRUConfig, device: torch.device,
            *, smoke: bool = False) -> tuple[GRUWinner, dict]:
    """Select a checkpoint ONLY by validation BCE. Test is never iterated."""
    if len(prepared.datasets["train"]) == 0 or len(prepared.datasets["val"]) == 0:
        raise ValueError("Train and validation both need sequence windows")
    seed = int(prepared.partitions.seed)
    seed_everything(seed)
    model = GRUWinner(prepared.features_sorted.shape[1], config.hidden_size,
                      config.num_layers, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    loaders = make_loaders(prepared, batch_size=config.batch_size,
                           num_workers=config.num_workers)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, min(2, config.max_epochs) + 1 if smoke else config.max_epochs + 1):
        train_loss = _epoch(model, loaders["train"], device, optimizer,
                            config.grad_clip, max_batches=2 if smoke else None)
        val_loss = _epoch(model, loaders["val"], device, None,
                          config.grad_clip, max_batches=2 if smoke else None)
        history.append({"epoch": epoch, "train_log_loss": train_loss,
                        "validation_log_loss": val_loss})
        if val_loss < best_loss - config.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("No valid GRU checkpoint was selected")
    model.load_state_dict(best_state)
    return model, {
        "schema": "sc2.sequential.gru_fit.v1",
        "status": "S4_SMOKE_PASS" if smoke else "S4_FIT_PASS",
        "seed": seed,
        "device": str(device),
        "config": asdict(config),
        "selected_epoch": best_epoch,
        "best_validation_log_loss": best_loss,
        "history": history,
        "train_windows": len(prepared.datasets["train"]),
        "validation_windows": len(prepared.datasets["val"]),
        "test_evaluation_policy": "not_accessed_during_training",
        "checkpoint_selection": "validation_bce_only",
        "smoke": smoke,
    }


@torch.no_grad()
def validation_predictions(model: GRUWinner, prepared: PreparedSequences,
                           index: IndexedSequences, split: PartitionedSequences,
                           device: torch.device, batch_size: int) -> tuple[pd.DataFrame, dict]:
    """Return predictions on the complete validation partition, in endpoint order."""
    dataset = prepared.datasets["val"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    model.eval()
    probabilities = []
    labels = []
    for x, y in loader:
        output = torch.sigmoid(model(x.to(device, non_blocking=True)))
        probabilities.append(output.cpu().numpy())
        labels.append(y.numpy())
    if not probabilities:
        raise ValueError("Validation partition contains no windows")
    probs = np.concatenate(probabilities).astype(np.float64)
    true = np.concatenate(labels).astype(np.int8)
    frame = pd.DataFrame({
        "replay_id": dataset.replay_ids,
        "time_sec": dataset.end_times.astype(np.int64),
        "y_true": true,
        "y_prob": probs,
        "y_pred": (probs >= 0.5).astype(np.int8),
    })
    if (not np.isfinite(probs).all() or (probs < 0).any() or
            (probs > 1).any() or frame.duplicated(["replay_id", "time_sec"]).any()):
        raise ValueError("Invalid GRU probabilities or duplicate endpoints")
    expected = endpoint_frame(index, split, "val")
    observed = frame.sort_values(["replay_id", "time_sec"], kind="stable").reset_index(drop=True)
    if (len(observed) != len(expected) or
            not np.array_equal(observed["replay_id"].to_numpy(), expected["replay_id"].to_numpy()) or
            not np.array_equal(observed["time_sec"].to_numpy(), expected["time_sec"].to_numpy()) or
            not np.array_equal(observed["y_true"].to_numpy(), expected["p1_wins"].to_numpy())):
        raise ValueError("GRU validation predictions do not match S1/S3 endpoints")
    y = observed["y_true"].to_numpy(dtype=np.int8)
    p = observed["y_prob"].to_numpy(dtype=np.float64)
    pred = observed["y_pred"].to_numpy(dtype=np.int8)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "n_endpoints": len(observed),
        "n_replays": int(observed["replay_id"].nunique()),
    }
    return observed, metrics
