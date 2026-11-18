"""S13: replay-stratified dynamic cutoff sampling at fixed S9 epoch budget.

Start from S12's immutable 4-anchor + 12-extra *train* endpoint pool.
Draw one cutoff from each of four anchor-centered time strata per replay
per epoch. Thus every epoch has exactly the original S9 4N examples.
Only the selection changes; S9/S12 saved outputs remain read-only.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict
from math import ceil

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .gru import GRUConfig, seed_everything
from .prefix import (
    CheckpointSequenceDataset, PackedGRUWinner, _epoch, _make_loader,
    collate_checkpoints,
)
from .progress import CHECKPOINTS


class DynamicReplayCutoffSampler(Sampler[list[int]]):
    """Exactly four endpoint examples per replay at each deterministic epoch.

    Anchor-centered strata are defined by the midpoint between neighboring
    observed-snapshot counts at 25/50/75/90% TRAIN anchors. The membership
    of these strata is fixed using training data only; each epoch samples one
    uniformly random endpoint per stratum per training replay.
    """

    def __init__(self, dataset: CheckpointSequenceDataset,
                 batch_size: int, seed: int):
        if batch_size < 1 or dataset.mode not in ("last9", "prefix"):
            raise ValueError("S13 requires a valid training dataset and batch size")
        rows = dataset.rows
        required = {"replay_id", "time_sec", "p1_wins",
                    "progress_percent", "observed_snapshot_count"}
        if not required.issubset(rows) or rows.empty or rows[list(required)].isna().any().any():
            raise ValueError("S13 training pool missing metadata")
        if rows.duplicated(["replay_id", "time_sec"]).any():
            raise ValueError("S13 repeated replay/time in training pool")
        if not set(rows.progress_percent).issubset({-1, *CHECKPOINTS}):
            raise ValueError("S13 training pool has an unknown checkpoint label")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.strata = []
        self.anchor_positions = []
        for _, group in rows.groupby("replay_id", sort=True, observed=True):
            if group.p1_wins.nunique() != 1:
                raise ValueError("S13 replay has contradictory winner labels")
            anchors = group.loc[
                group.progress_percent.isin(CHECKPOINTS)
            ].sort_values("progress_percent", kind="stable")
            if len(anchors) != 4 or anchors.progress_percent.tolist() != list(CHECKPOINTS):
                raise ValueError("S13 requires four original training anchors per replay")
            anchor_counts = anchors.observed_snapshot_count.to_numpy(dtype=np.int64)
            if not np.all(np.diff(anchor_counts) > 0):
                raise ValueError("S13 train anchor snapshot counts must increase")
            midpoint = (anchor_counts[:-1].astype(np.float64)
                        + anchor_counts[1:].astype(np.float64)) / 2.0
            counts = group.observed_snapshot_count.to_numpy(dtype=np.int64)
            if not np.all((counts >= anchor_counts[0]) & (counts <= anchor_counts[-1])):
                raise ValueError("S13 extra cutoff outside original anchor time range")
            region = np.searchsorted(midpoint, counts, side="right")
            locations = group.index.to_numpy(dtype=np.int64)
            one_replay = []
            for i in range(4):
                pool = locations[region == i]
                if len(pool) < 1 or int(anchors.index[i]) not in pool:
                    raise ValueError("S13 missing anchor in a cutoff region")
                one_replay.append(pool)
            self.strata.append(tuple(one_replay))
            self.anchor_positions.extend(anchors.index.to_numpy(dtype=np.int64).tolist())
        self.n_replays = len(self.strata)
        self.samples_per_epoch = self.n_replays * len(CHECKPOINTS)
        if self.samples_per_epoch < 1 or len(self.anchor_positions) != self.samples_per_epoch:
            raise ValueError("S13 invalid training replay population")
        if self.samples_per_epoch > len(dataset):
            raise ValueError("S13 epoch selection larger than available training pool")

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("S13 epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return ceil(self.samples_per_epoch / self.batch_size)

    def selected_indices(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 1009 * self.epoch)
        indices = [
            int(pool[rng.integers(0, len(pool))])
            for four in self.strata for pool in four
        ]
        result = np.asarray(indices, dtype=np.int64)
        if len(result) != self.samples_per_epoch or len(np.unique(result)) != len(result):
            raise ValueError("S13 dynamic train selection is not replay-unique")
        return result

    def __iter__(self):
        selected = self.selected_indices()
        rng = np.random.default_rng(self.seed + 1009 * self.epoch + 0x5A17)
        rng.shuffle(selected)
        batches = []
        block_size = self.batch_size * 16
        for offset in range(0, len(selected), block_size):
            block = selected[offset:offset + block_size]
            block = block[np.argsort(self.dataset.lengths[block], kind="stable")]
            for start in range(0, len(block), self.batch_size):
                batches.append(block[start:start + self.batch_size].tolist())
        rng.shuffle(batches)
        if len(batches) != len(self):
            raise ValueError("S13 loader batch budget differs from S9")
        yield from batches


def fit_dynamic_gru(train: CheckpointSequenceDataset,
                    val: CheckpointSequenceDataset,
                    config: GRUConfig, device: torch.device,
                    seed: int, *, smoke: bool = False, verbose: bool = False):
    """Reuse S9 architecture, optimizer, validation, BCE and early-stop policy."""
    if train.mode != val.mode or train.values.shape[1] != val.values.shape[1]:
        raise ValueError("S13 train/validation GRU mode or features mismatch")
    if len(val.rows) != 4 * val.rows.replay_id.nunique():
        raise ValueError("S13 validation must remain four endpoints per replay")
    seed_everything(seed)
    model = PackedGRUWinner(train.values.shape[1], config).to(device)
    sampler = DynamicReplayCutoffSampler(train, config.batch_size, seed)
    training_loader = DataLoader(
        train, batch_sampler=sampler, collate_fn=collate_checkpoints,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader, _ = _make_loader(
        val, config.batch_size, seed, False, config.num_workers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    best_state = None
    best_bce = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    max_epochs = min(2, config.max_epochs) if smoke else config.max_epochs
    for epoch in range(1, max_epochs + 1):
        sampler.set_epoch(epoch)
        selection = sampler.selected_indices()
        selected_original_anchors = int(np.isin(
            selection, sampler.anchor_positions,
        ).sum())
        digest = hashlib.sha256(
            np.sort(selection).astype("<i8", copy=False).tobytes()
        ).hexdigest()
        train_bce = _epoch(
            model, training_loader, device, optimizer, config.grad_clip,
            max_batches=2 if smoke else None,
        )
        val_bce = _epoch(
            model, val_loader, device, None, config.grad_clip,
            max_batches=2 if smoke else None,
        )
        if not np.isfinite(train_bce) or not np.isfinite(val_bce):
            raise ValueError("S13 nonfinite training or validation BCE")
        history.append({
            "epoch": epoch,
            "train_log_loss": float(train_bce),
            "validation_log_loss": float(val_bce),
            "epoch_train_examples": sampler.samples_per_epoch,
            "selected_original_anchors": selected_original_anchors,
            "selected_indices_sha256": digest,
        })
        if verbose:
            print(
                f"[S13 {train.mode}] epoch={epoch}/{max_epochs} "
                f"train_BCE={train_bce:.6f} val_BCE={val_bce:.6f} "
                f"train_cutoffs={sampler.samples_per_epoch} "
                f"anchors={selected_original_anchors}",
                flush=True,
            )
        if val_bce < best_bce - config.min_delta:
            best_bce, best_epoch, stale = val_bce, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("S13 could not select a validation-best checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "schema": "sc2.sequential.dynamic_gru_fit.v1",
        "status": "S13_SMOKE_FIT_PASS" if smoke else "S13_FIT_PASS",
        "mode": train.mode,
        "seed": int(seed),
        "device": str(device),
        "config": asdict(config),
        "selected_epoch": int(best_epoch),
        "best_validation_log_loss": float(best_bce),
        "history": history,
        "train_pool_examples": len(train),
        "train_replays": sampler.n_replays,
        "train_examples_per_epoch": sampler.samples_per_epoch,
        "train_batches_per_epoch": len(sampler),
        "validation_examples": len(val),
        "train_length_min": int(train.lengths.min()),
        "train_length_max": int(train.lengths.max()),
        "validation_length_min": int(val.lengths.min()),
        "validation_length_max": int(val.lengths.max()),
        "checkpoint_selection": "validation_bce_only",
        "test_evaluation_policy": "not_accessed_during_training",
        "sampler": "four_anchor_midpoint_strata_one_train_cutoff_per_stratum_per_replay_per_epoch",
        "smoke": bool(smoke),
    }
