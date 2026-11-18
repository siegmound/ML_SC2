"""S9: matched endpoint training for short-9 and entire-observed-prefix GRUs.

Both models use identical train/validation replay cohorts and labels.
Only sequence start differs. Checkpoint percentages and eventual replay
length are used to select EXAMPLES offline, never included in model inputs.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from math import ceil
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from .comparison import model_metrics, paired_replay_bootstrap
from .data import IndexedSequences
from .gru import GRUConfig, seed_everything
from .progress import CHECKPOINTS

Mode = Literal["last9", "prefix"]
MODES = ("last9", "prefix")


class CheckpointSequenceDataset(Dataset):
    """Indexes a shared normalized [snapshot,feature] array, without copying it."""

    def __init__(self, values: np.ndarray, index: IndexedSequences,
                 checkpoint_rows: pd.DataFrame, mode: Mode):
        if mode not in MODES:
            raise ValueError("Invalid checkpoint sequence mode")
        if values.dtype != np.float32 or values.ndim != 2 or not values.flags.c_contiguous:
            raise ValueError("Sequence storage must be contiguous float32")
        if values.shape[0] != len(index.ordered_rows):
            raise ValueError("Snapshot matrix and source index do not match")
        if checkpoint_rows.empty:
            raise ValueError("Empty training/validation checkpoints")
        if checkpoint_rows.duplicated(["replay_id", "time_sec"]).any():
            raise ValueError("Repeated checkpoint prediction times")
        w = index.windows
        keys = pd.MultiIndex.from_frame(w[["replay_id", "time_sec"]])
        if not keys.is_unique:
            raise ValueError("Nonunique window index")
        requests = pd.MultiIndex.from_frame(checkpoint_rows[["replay_id", "time_sec"]])
        positions = keys.get_indexer(requests)
        if (positions < 0).any():
            raise ValueError("S9 endpoint missing from frozen nine-step index")
        matched = w.iloc[positions].reset_index(drop=True)
        selected = checkpoint_rows.reset_index(drop=True)
        if not np.array_equal(matched["p1_wins"].to_numpy(dtype=np.int8),
                              selected["p1_wins"].to_numpy(dtype=np.int8)):
            raise ValueError("S9 target mismatch at requested checkpoint")
        ends = matched["stop_row"].to_numpy(dtype=np.int64)
        last9_starts = matched["start_row"].to_numpy(dtype=np.int64)
        replay_start = index.ordered_rows.reset_index().groupby(
            "replay_id", sort=False, observed=True)["index"].min()
        firsts = selected["replay_id"].map(replay_start).to_numpy(dtype=np.int64)
        lengths_prefix = ends - firsts
        observed = selected["observed_snapshot_count"].to_numpy(dtype=np.int64)
        if not np.array_equal(lengths_prefix, observed):
            raise ValueError("S9 prefix crosses a replay boundary or wrong observation count")
        if not (ends > firsts).all() or not np.all(ends - last9_starts == 9):
            raise ValueError("S9 checkpoint window has invalid bounds")
        self.values = values
        self.rows = selected.copy()
        self.starts = last9_starts if mode == "last9" else firsts
        self.ends = ends
        self.lengths = self.ends - self.starts
        self.labels = selected["p1_wins"].to_numpy(dtype=np.float32)
        self.mode = mode
        if (self.lengths < 1).any() or (self.lengths > len(values)).any():
            raise ValueError("Invalid dataset sequence lengths")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.values[self.starts[i]:self.ends[i]]), float(self.labels[i]), i


def collate_checkpoints(batch):
    if not batch:
        raise ValueError("Empty sequence batch")
    lengths = torch.tensor([len(s[0]) for s in batch], dtype=torch.int64)
    if (lengths < 1).any():
        raise ValueError("Sequence cannot have zero observations")
    feats = len(batch[0][0][0])
    padded = torch.zeros((len(batch), int(lengths.max().item()), feats), dtype=torch.float32)
    for i, (array, _, _) in enumerate(batch):
        padded[i, :len(array)] = array
    labels = torch.tensor([s[1] for s in batch], dtype=torch.float32)
    indices = np.array([s[2] for s in batch], dtype=np.int64)
    return padded, lengths, labels, indices


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle all examples, then group nearby lengths to reduce padding overhead."""

    def __init__(self, lengths: np.ndarray, batch_size: int, seed: int,
                 *, shuffle: bool):
        if batch_size < 1 or not len(lengths) or (lengths < 1).any():
            raise ValueError("Invalid bucketing parameters")
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1009 * self.epoch)
        indices = np.arange(len(self.lengths))
        if self.shuffle:
            rng.shuffle(indices)
        groups = []
        size = self.batch_size * 16
        for offset in range(0, len(indices), size):
            block = indices[offset:offset + size]
            block = block[np.argsort(self.lengths[block], kind="stable")]
            for start in range(0, len(block), self.batch_size):
                groups.append(block[start:start + self.batch_size].tolist())
        if self.shuffle:
            rng.shuffle(groups)
        yield from groups


class PackedGRUWinner(nn.Module):
    """Architecture-identical to S4 except length-aware final-state extraction."""

    def __init__(self, features: int, config: GRUConfig):
        super().__init__()
        if features < 1:
            raise ValueError("Invalid input feature count")
        self.features = features
        self.gru = nn.GRU(features, config.hidden_size, config.num_layers,
                          batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        if x.ndim != 3 or x.shape[2] != self.features or len(lengths) != len(x):
            raise ValueError("Invalid model input shape")
        if (lengths < 1).any() or (lengths > x.shape[1]).any():
            raise ValueError("Invalid packed sequence lengths")
        packed = pack_padded_sequence(x, lengths.to("cpu"),
                                      batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(hidden[-1]).squeeze(-1)


def _make_loader(dataset, batch_size: int, seed: int, shuffle: bool,
                 num_workers: int):
    sampler = LengthBucketBatchSampler(dataset.lengths, batch_size, seed,
                                        shuffle=shuffle)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_checkpoints,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader, sampler


def _epoch(model, loader, device, optimizer, grad_clip: float,
           max_batches: int | None = None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    loss_total = 0.0
    count = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (x, lengths, y, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            if is_train:
                (loss / len(y)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            loss_total += float(loss.detach().item())
            count += len(y)
    if count == 0:
        raise ValueError("No train/validation batches")
    return loss_total / count


def fit_checkpoint_gru(train: CheckpointSequenceDataset,
                       val: CheckpointSequenceDataset, config: GRUConfig,
                       device: torch.device, seed: int, *,
                       smoke: bool = False, verbose: bool = False):
    """Train-only optimizer and validation-selected checkpoint, never iterates test."""
    if train.mode != val.mode:
        raise ValueError("Train/validation must share the same sequence mode")
    if train.values.shape[1] != val.values.shape[1]:
        raise ValueError("Feature count mismatch")
    seed_everything(seed)
    model = PackedGRUWinner(train.values.shape[1], config).to(device)
    train_loader, train_sampler = _make_loader(
        train, config.batch_size, seed, True, config.num_workers)
    val_loader, _ = _make_loader(val, config.batch_size, seed, False, config.num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)
    history = []
    best_state = None
    best = float("inf")
    best_epoch = 0
    stale = 0
    limit = min(2, config.max_epochs) if smoke else config.max_epochs
    for epoch in range(1, limit + 1):
        train_sampler.set_epoch(epoch)
        loss_train = _epoch(model, train_loader, device, opt, config.grad_clip,
                            max_batches=2 if smoke else None)
        loss_val = _epoch(model, val_loader, device, None, config.grad_clip,
                          max_batches=2 if smoke else None)
        history.append({"epoch": epoch, "train_log_loss": loss_train,
                        "validation_log_loss": loss_val})
        if verbose:
            print(f"[S9 {train.mode}] epoch={epoch}/{limit} "
                  f"train_BCE={loss_train:.6f} val_BCE={loss_val:.6f}", flush=True)
        if loss_val < best - config.min_delta:
            best, best_epoch, stale = loss_val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("No valid validation-selected checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "schema": "sc2.sequential.packed_gru_fit.v1",
        "status": "S9_SMOKE_PASS" if smoke else "S9_FIT_PASS",
        "mode": train.mode, "seed": seed, "device": str(device),
        "config": asdict(config), "selected_epoch": best_epoch,
        "best_validation_log_loss": best, "history": history,
        "train_examples": len(train), "validation_examples": len(val),
        "train_length_min": int(train.lengths.min()),
        "train_length_max": int(train.lengths.max()),
        "val_length_min": int(val.lengths.min()),
        "val_length_max": int(val.lengths.max()),
        "checkpoint_selection": "validation_bce_only",
        "test_evaluation_policy": "not_accessed_during_training",
        "smoke": smoke,
    }


@torch.no_grad()
def predict_checkpoint_gru(model: PackedGRUWinner,
                           dataset: CheckpointSequenceDataset,
                           config: GRUConfig, device: torch.device):
    loader, _ = _make_loader(dataset, config.batch_size, 0, False, 0)
    model.eval()
    probs = np.full(len(dataset), np.nan, dtype=np.float64)
    for x, lengths, _, positions in loader:
        logits = model(x.to(device, non_blocking=True), lengths)
        probs[positions] = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("Non-finite S9 validation probability or incomplete coverage")
    out = dataset.rows.loc[:, ["replay_id", "time_sec", "p1_wins",
                                "progress_percent"]].copy()
    out["y_prob"] = probs
    out["y_pred"] = (probs >= 0.5).astype(np.int8)
    out["y_true"] = out.pop("p1_wins").astype(np.int8)
    return out


def compare_modes(predictions: dict[str, pd.DataFrame], *,
                  reps: int, seed: int):
    """One prediction per checkpoint/replay, exact matched keys, per-progress CI."""
    if set(predictions) != set(MODES) or reps < 0:
        raise ValueError("S9 requires both trained modes and nonnegative reps")
    a, b = (predictions[mode].sort_values(
        ["replay_id", "progress_percent"], kind="stable").reset_index(drop=True)
            for mode in MODES)
    keys = ("replay_id", "time_sec", "progress_percent", "y_true")
    for k in keys:
        if not np.array_equal(a[k].to_numpy(), b[k].to_numpy()):
            raise ValueError(f"S9 mismatched paired checkpoint {k}")
    if len(a) != 4 * a.replay_id.nunique() or a.duplicated(
        ["replay_id", "progress_percent"]
    ).any():
        raise ValueError("S9 checkpoint population is not balanced")
    result = {}
    for i, progress in enumerate(CHECKPOINTS):
        ma = a[a.progress_percent == progress].reset_index(drop=True)
        mb = b[b.progress_percent == progress].reset_index(drop=True)
        if len(ma) != a.replay_id.nunique():
            raise ValueError("S9 progress is missing a replay")
        y = ma.y_true.to_numpy(dtype=np.int8)
        scores = {
            "last9": model_metrics(y, ma.y_prob.to_numpy(dtype=np.float64),
                                   ma.y_pred.to_numpy(dtype=np.int8)),
            "prefix": model_metrics(y, mb.y_prob.to_numpy(dtype=np.float64),
                                    mb.y_pred.to_numpy(dtype=np.int8)),
        }
        paired = pd.DataFrame({
            "replay_id": ma.replay_id,
            "p1_wins": y,
            "v31_prob": ma.y_prob,
            "v31_pred": ma.y_pred,
            "gru_prob": mb.y_prob,
            "gru_pred": mb.y_pred,
        })
        interval = paired_replay_bootstrap(
            paired, "v31", reps=reps, seed=seed + i)
        delta = {m: scores["prefix"][m] - scores["last9"][m]
                 for m in ("accuracy", "balanced_accuracy", "roc_auc", "log_loss",
                           "brier_score")}
        if not np.isclose(delta["accuracy"], interval["accuracy_delta"], atol=1e-12,
                          rtol=0) or not np.isclose(delta["log_loss"],
                                                   interval["log_loss_delta"],
                                                   atol=1e-9, rtol=0):
            raise ValueError("S9 paired interval and point estimates disagree")
        result[str(progress)] = {
            "n_replays": len(ma), "models": scores,
            "delta_prefix_minus_last9": delta,
            "paired_replay_bootstrap": interval,
        }
    return result
