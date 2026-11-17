# S1 — Replay-safe sequence index

**Base:** `feature/sc2-sequential-models`, derived from `audit/sc2-corrections-2026-11-17`.
**Scope:** read-only construction of fixed-size temporal windows from the frozen `v3_1_fixed` CSV ZIP. **No training, no split-aware preprocessing, no historical artifact changes.**

## Data contract

- Input ZIP: `data/processed/starcraft_full_dataset_v3_1_fixed.zip`. Requires a real Git LFS payload, not a Git LFS pointer. The source must contain exactly one CSV.
- Schema: exact **ordered** 64 feature names from `data/processed/manifests/b3_feature_lineage_v1.json`. `replay_id`, `time_sec`, `p1_wins` are never features.
- Metadata: `replay_id` nonempty, unique `(replay_id,time_sec)`, nonnegative 15-second grid, `p1_wins` binary and constant per replay.
- Windows: `[start_row,stop_row)` offsets into the **sorted** `ordered_rows.parquet`, where `stop_row` is exclusive. Every window uses L rows within a single replay, with consecutive `time_sec` values 15 seconds apart. Gaps split contiguous runs.
- `ordered_rows.parquet` contains `source_row`: the original physical row index in the CSV. This is essential when an S2 loader retrieves feature values and when the S1 CLI uses `--max-replays` for smoke testing.
- All windows end at `time_sec` and use no future snapshots. They are generated globally within replay in S1. **Train/validation/test group partitioning and train-only normalization are implemented in S2, not here.**
- S1 writes only new files beneath `results/sequential/`, refuses an existing output directory and records the actual input ZIP SHA-256 plus feature ordering in a JSON manifest. Outputs are gitignored.

## Expected full v3.1 counts (window stride 1)

| Length | History covered | Windows | Replays with windows |
|---|---|---:|---:|
| 9 | 120 seconds | 706,952 | 22,700 |
| 17 | 240 seconds | 530,870 | 20,834 |
| 33 | 480 seconds | 250,457 | 12,910 |

These counts were independently checked against the intact frozen v3.1 CSV ZIP. The total source dataset has 892,047 rows from 23,241 replays. Counts will differ when limiting replays for a smoke test.

## FAIR tests and index generation

From the repository root, with the `ml-sc2` venv active:

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
python scripts/70_verify_freeze.py --mode pointer
python scripts/58_verify_final_table.py --check

# Without any writes: full-data checks and window counts.
python scripts/74_build_sequence_index.py --window 9 --dry-run
python scripts/74_build_sequence_index.py --window 17 --dry-run
python scripts/74_build_sequence_index.py --window 33 --dry-run

# Small on-disk smoke test, 10 deterministically chosen replays.
python scripts/74_build_sequence_index.py --window 9 --max-replays 10 \
  --output-dir results/sequential/s1_smoke_9

# Real index: 706,952 windows. Will not overwrite existing output.
python scripts/74_build_sequence_index.py --window 9 \
  --output-dir results/sequential/v3_1_fixed_w9_s1
```

To inspect the results after the last command:

```bash
python - <<'PY'
import json
from pathlib import Path
import pandas as pd
p = Path('results/sequential/v3_1_fixed_w9_s1')
m = json.loads((p / 'manifest.json').read_text())
w = pd.read_parquet(p / 'window_index.parquet')
r = pd.read_parquet(p / 'ordered_rows.parquet')
assert len(w) == m['windows'] == 706952
assert len(r) == m['rows'] == 892047
assert (w['stop_row'] - w['start_row'] == 9).all()
print('S1 full index PASS', m['windows'], 'windows', m['windowed_replays'], 'replays')
PY
```

To download and verify the frozen v3.1 LFS payload:

```bash
git lfs pull --include='data/processed/starcraft_full_dataset_v3_1_fixed.zip' --exclude=''
sha256sum data/processed/starcraft_full_dataset_v3_1_fixed.zip
```

The expected B0 SHA-256 is `65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517`. Note that `scripts/70_verify_freeze.py --mode payload` verifies **both** v3.1 and v3.2, and will fail if v3.2 has not been downloaded.

**Do not run historical RF/XGB/deep training scripts in S1.** S1 builds no new trained model or published classification metrics.

### FAIR troubleshooting: Git LFS is not installed

A fresh clone can leave a ~134-byte LFS pointer at the dataset path when `git-lfs` is unavailable. A pointer is **not** a corrupt dataset, and `--dry-run` still requires the real ZIP because it inspects the CSV schema and counts actual windows.

On Debian/Ubuntu machines **with sudo permission**:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install --local
git lfs pull --include='data/processed/starcraft_full_dataset_v3_1_fixed.zip' --exclude=''
sha256sum data/processed/starcraft_full_dataset_v3_1_fixed.zip
```

If the shared HPC workstation does not permit `sudo`, ask the administrator for Git LFS or install the `git-lfs` conda-forge binary in a user-controlled environment and ensure its `bin` directory is on `PATH` for the same shell before running the `git lfs` commands. Do not modify the historical ZIP manually, and do not commit downloaded dataset payloads.

The SHA-256 of the **real** v3.1 archive must match the B0 manifest above. Only then retry the three `--dry-run` checks and smoke/full index construction.
