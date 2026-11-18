# S2 — Replay splits, train-only preprocessing, PyTorch Dataset

**Branch:** `feature/sc2-sequential-models`. S1 validated on FAIR: 23,241 replay groups and 706,952 nine-snapshot windows, 64 features. S2 adds no new ML training, test predictions or historical artifact edits.

## Design contract

1. Consume S1 `ordered_rows.parquet`, `window_index.parquet`, `manifest.json`; the full v3.1 LFS ZIP remains the immutable feature source. Check the source ZIP SHA-256 against the S1 manifest before reading feature values.
2. Read the exact ordered S1 predictor list, and permute physical CSV feature rows using `ordered_rows.source_row`. Window offsets are **into S1 sorted rows**, not original CSV row positions.
3. Load the historical seed-specific `train_groups`, `val_groups`, `test_groups`. Fail on duplicate or overlapping group assignments. The full index must cover exactly the original groups. A deliberately partial S1 `--max-replays` smoke index may cover a strict subset but cannot contain unknown groups.
4. Assign complete windows to a single replay partition. Verify every window remains in one replay and one uninterrupted 15-second run, with the label and timestamp taken from its last row.
5. Fit `StandardScaler` **only on unique training snapshot rows**, not the overlapping windows, nor validation/test rows. Apply the learned transform to all three partitions. Store one shared contiguous float32 feature matrix and slice windows lazily (`Dataset.__getitem__`), no materialized [N,L,F] tensor on disk.
6. The `DataLoader` emits `(X, y)` with X.shape `[batch, L, 64]`, y.shape `[batch]`, both float32. Defaults: shuffle train only, deterministic training shuffle seed from split JSON, 0 workers, no model training.
7. S2 is a data-integrity/loader gate only. Test-set outcome metrics will **not** be computed in S2.

## FAIR run (from project root)

```bash
source "$HOME/.venvs/ml-sc2/bin/activate"
git pull --ff-only
python -m unittest discover -s tests -p 'test_*.py' -q
python scripts/70_verify_freeze.py --mode pointer
python scripts/58_verify_final_table.py --check

python scripts/75_check_sequence_loader.py --batch-size 32
python scripts/75_check_sequence_loader.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_43.json --batch-size 32
python scripts/75_check_sequence_loader.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_44.json --batch-size 32
```

Each run should print `"status": "S2_LOADER_PASS"`; in seed 42 the reported `scaler_fitted_rows` is expected to be 571,428 snapshot rows (all training replay snapshots). The combined window counts must total 706,952 per seed. `source_sha256` must equal `65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517`.

Run metadata only; neither this command nor the module creates model checkpoints or reports test Accuracy. S2 does not silently overwrite any historical results.

## Notes for S3–S4

- Endpoint-matched comparisons must use `(replay_id, time_sec)` from each `SequenceWindowDataset.replay_ids` and `end_times`.
- When comparing sequence lengths 9/17/33, restrict models to common replay/time endpoints.
- Sequences within the same replay are highly correlated; split and statistical uncertainty must be group-aware.
- Historical feature-lineage gaps and the already-inspected test holdout remain limitations.
