# SC2 V4: independent clean-room rebuild protocol

Status: INITIAL IMPLEMENTATION, synthetic CI only. Raw FAIR replay build and
model reruns have NOT yet been executed. This is NOT historical v3.1/v3.2
raw-to-final parity and does not unblock historical B3 or B5 release gates.

## Scope and explicit validity limits

The source is a NEW raw .SC2Replay directory, not a historical processed
CSV. A replay's SHA-256 of binary content becomes its replay_id so byte
duplicates across tournament packs cannot leak between group splits.
Every source path, SHA-256, accepted replay and exclusion reason goes into
a new per-replay audit CSV. The builder refuses to overwrite older outputs,
historical data or a previous V4 report.

The V4 tracker interpretation deduplicates the same unit_id on UnitInit and
UnitBorn, handles unit type changes/deaths and makes qualitative counter
direction explicit. UNIT_COUNTERS[target] denotes effective OPPONENT
counters against the target: own combat penalties use the direct list,
own counter advantage uses the reversed direction. Numeric coefficients
are prespecified project heuristics, NOT recovered Blizzard mechanics
or coefficients selected using historical test outcomes.

The primary CORE feature profile drops all feature names containing
combat or counter (including temporal derivatives). FULL is an exploratory
profile containing those heuristic scores. Choose core as primary before
seeing any new test metrics; full is an ablation only.

Data source limitations: currently FAIR has two SHA-identical copies of
300 named raw entries, of which 295 are MPQ candidates and five are
AppleDouble metadata. This is not the 24,057-replay original corpus;
downloading/scanning all 71 SC2ReSet replaypacks remains a separate
future data acquisition stage. Actual input count and included replay
count for V4 must be discovered by running V4 against real raw files.

The model runners are reused with fresh input data: RF and XGB run
group-disjoint training, tuning, calibration and final test; the MLP
selects architecture by replay-disjoint validation and evaluates only
the selected model on test. Seeds 42/43/44 are stability checks, NOT
three independent previously unseen test sets. The test sets might
contain replays from the previously inspected 3-replay audit; no claim
of pristine blindness is authorized.

## FAIR commands: synthetic tests

Run from the existing project checkout; do not discard local changes.

    cd ~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean
    git status --short
    git fetch origin rebuild/sc2-v4-clean-2026-11-20
    git switch --track origin/rebuild/sc2-v4-clean-2026-11-20
    export SC2_PYTHON="$HOME/.venvs/ml-sc2/bin/python"
    export PYTHONPATH="$PWD/src"
    "$SC2_PYTHON" -m unittest discover -s tests -p 'test_v4_clean_rebuild.py' -v
    "$SC2_PYTHON" -m unittest discover -s tests -p 'test_v4_suite.py' -v

If that branch already exists locally, instead switch normally and
pull --ff-only from origin. Do not reset, clean or switch away from
uncommitted scientific artifacts.

## Step 1: small raw replay build, 120 SHA-ordered input identities

    "$SC2_PYTHON" scripts/90_v4_build_dataset.py \
      --raw-root data/raw/replay_subsets \
      --max-replays 120 \
      --output-dir results/v4_raw_smoke120_20261120

    cat results/v4_raw_smoke120_20261120/manifest.json
    head -n 20 results/v4_raw_smoke120_20261120/per_replay_source_audit.csv

The output has a SHA-256-pinned data ZIP, reproducible feature headers,
full per-source file ledger and machine-readable exclusion reason counts.
If fewer than 40 unique replay identities are accepted, create a NEW
output directory for a larger build (up to all available entries).

## Step 2: single-seed RF and XGB CPU protocol smoke

    "$SC2_PYTHON" scripts/91_v4_run_suite.py \
      --dataset-dir results/v4_raw_smoke120_20261120 \
      --output-root results/v4_experiments_smoke42_20261120 \
      --stage smoke --feature-profile core \
      --models rf xgb --seeds 42 --device cpu --n-jobs 4 --plan

Review the plan. Then run the same command replacing --plan with
--execute. It refuses to continue or overwrite an existing run directory;
when troubleshooting failures, preserve logs and choose a NEW root.

## Step 3: all locally available MPQ games, three models / three seeds

    "$SC2_PYTHON" scripts/90_v4_build_dataset.py \
      --raw-root data/raw/replay_subsets \
      --output-dir results/v4_raw_all_available_20261120

    "$SC2_PYTHON" scripts/91_v4_run_suite.py \
      --dataset-dir results/v4_raw_all_available_20261120 \
      --output-root results/v4_experiments_all_available_20261120 \
      --stage full --feature-profile core \
      --models rf xgb deep --seeds 42 43 44 \
      --device cuda --n-jobs 8 --plan

After validating source coverage and GPU, rerun with --execute to
train nine independent model/seed configurations, preserving all logs.
The full stage means all CURRENTLY AVAILABLE raw replay inputs,
not the historical 23,241/24,057 game corpus.

To compare an exploratory FULL feature profile, use a different
output-root with the same seed selection, selecting --feature-profile
full. Report core vs full as prespecified feature ablation. Do not
choose the winner by looking at the test scores.

## Required gates before calling V4 correct or publishable

1. Confirm real FAIR raw MPQ parsing/metadata and all exclusion reasons.
2. Audit the tracker unit lifecycle and at least a few detailed replay
   event traces against sc2reader; synthetic tests alone are insufficient.
3. Check duplicate match content across packs, including near-duplicate
   replays whose binary files have different SHA-256 hashes.
4. Verify code/dependency versions, generated ZIP hashes, split
   membership, all prediction rows/metrics, no test-driven selection.
5. Report per-snapshot and per-replay metrics; snapshots from one match
   are correlated and cannot supply naive independent confidence bounds.
6. For full-scale scientific comparability, obtain a source-verified
   complete raw replay corpus from SC2ReSet and rebuild anew. A new
   final count need not match historical counts without matching filters.

Historical frozen datasets, B2 runs and original paper/presentation stay
unchanged; historical B3 false and B5 BLOCKED_PENDING_EVIDENCE.


## Step 4: verify saved test predictions and per-replay metrics

Once a suite has finished, verify that each test prediction belongs
to the saved held-out source replay/time key, has the correct game
winner label, has a finite probability and reproduces every reported
test metric. Compute additional **per-replay** metrics by averaging
the model's snapshot probabilities within each test replay (each
replay counts once). These measures should be reported alongside,
rather than replaced by, ordinary per-snapshot accuracy.

For the completed initial RF+XGB smoke suite:

    "$SC2_PYTHON" scripts/92_v4_verify_results.py \
      --dataset-dir results/v4_raw_smoke120_20261120 \
      --suite-root results/v4_experiments_smoke42_20261120 \
      --output-dir results/v4_verification_smoke42_20261120

    cat results/v4_verification_smoke42_20261120/summary.csv

The verifier fails closed if any planned model/seed output is absent,
any prediction key or label differs from the raw v4 dataset, split
groups overlap, output report and saved predictions disagree, or
the source/profile SHA-256 changes. It creates a NEW output folder
and cannot overwrite earlier evidence. It is not a model selection
or hyperparameter tuning step.

For the all-available-data nine-model-seed suite, use the matching
dataset/suite paths and a NEW verification output directory.

## Original raw archive fully recovered: 24,057 replay files

FAIR now contains the original outer archive at
`/home/jlpfritas/HPC-POMDP/ML-SC2/replays.zip`, SHA-256
`8dcb826aeb17481d592f2d564b8a3af72bcb2f11fba9421bdbd350641970241d`.
The actual ZIP central directory has **24,057 direct `.SC2Replay`
members with distinct basenames**, **71 separate tournament ZIPs**
and **one JSON** (24,129 file entries total). The extra tournament
ZIPs are not to be extracted/double-counted. The original raw count
is now physically evidenced, but the source-v3.1 identity relation
and individual exclusion reasons remain to be verified.

### Read-only raw ZIP to frozen v3.1 ID reconciliation

To get the new scripts when currently on the old audit worktree:

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean
git fetch origin refs/heads/rebuild/sc2-v4-clean-2026-11-20:refs/remotes/origin/rebuild/sc2-v4-clean-2026-11-20
git switch -c rebuild/sc2-v4-clean-2026-11-20 --track origin/rebuild/sc2-v4-clean-2026-11-20
export SC2_PYTHON="$HOME/.venvs/ml-sc2/bin/python"
export PYTHONPATH="$PWD/src"
"$SC2_PYTHON" -m unittest discover -s tests -p 'test_b3_recovered_original_zip.py' -v
"$SC2_PYTHON" -m unittest discover -s tests -p 'test_v4_original_zip_extraction.py' -v

"$SC2_PYTHON" scripts/93_b3_reconcile_original_replays_zip.py \
  --raw-zip /home/jlpfritas/HPC-POMDP/ML-SC2/replays.zip \
  --frozen-v31-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip \
  --output-dir results/b3_recovered_full_zip_census_20261123

cat results/b3_recovered_full_zip_census_20261123/summary.json
head -n 16 results/b3_recovered_full_zip_census_20261123/raw_not_in_frozen.csv
grep '\.json,' results/b3_recovered_full_zip_census_20261123/outer_zip_other_members.csv
```

Script 93 checks BOTH source and frozen SHA-256, scans 24,057 replay
member names without decompressing replay bytes and reads only the
frozen CSV replay_id column in chunks. It reports source-only IDs
without guessing exclusion causes. The JSON top-level keys are
recorded to help identify provenance.

### Optional selective extraction only after census

```bash
"$SC2_PYTHON" scripts/94_v4_extract_original_replays.py \
  --archive /home/jlpfritas/HPC-POMDP/ML-SC2/replays.zip \
  --output-dir /home/jlpfritas/HPC-POMDP/ML-SC2/v4_original_replays_extracted \
  --plan
```

Repeat with `--extract` after reviewing the plan. It copies only
replay members into the NEW output root's `replays/` folder, checking
ZIP CRC and replay SHA-256 and preserving a source-member ledger.
The 71 ZIPs and original archive remain untouched. Then build an
independent raw V4 120-replay smoke with script 90 before attempting
the entire source corpus. Original frozen v3.1/v3.2 datasets are never
overwritten.

## Verified complete raw-to-frozen source-name reconciliation (2026-11-23)

FAIR executed `scripts/93_b3_reconcile_original_replays_zip.py` against
the SHA-pinned original ZIP (24,057 direct replay basenames) and frozen
v3.1 archive (892,047 rows, 23,241 replay IDs). **All 23,241 frozen
IDs are contained in the recovered raw ZIP and exactly 816 source
replay basenames are absent from frozen v3.1.** None of the frozen
IDs are absent from source. The actual 816 list is saved on FAIR at
`results/b3_recovered_full_zip_census_20261123/raw_not_in_frozen.csv`.

This proves an exact identity-level subset, NOT that the original
builder excluded all 816 for a particular reason, nor that the
current source code has historical feature parity. Historical B3
and B5 gates remain blocked.

### Protect the audit checkout with a separate V4 Git worktree

FAIR still holds many original B2/B3 evidence directories and a
locally modified `results/logs/reproducibility_check.txt`. Rather
than trying another checkout inside that worktree, create a separate
directory for V4 commands (Git worktree reuses the same repository
history but leaves audit files untouched):

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean
git fetch origin refs/heads/rebuild/sc2-v4-clean-2026-11-20:refs/remotes/origin/rebuild/sc2-v4-clean-2026-11-20
GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach \
  "$HOME/HPC-POMDP/ML-SC2/ML_SC2_v4_rebuild_clean" \
  origin/rebuild/sc2-v4-clean-2026-11-20
cd "$HOME/HPC-POMDP/ML-SC2/ML_SC2_v4_rebuild_clean"
export SC2_PYTHON="$HOME/.venvs/ml-sc2/bin/python"
export PYTHONPATH="$PWD/src"
```

Because the detached V4 checkout can contain Git LFS *pointers*
in place of frozen payloads, use the **verified absolute location**
from the existing audit worktree as the historical v3.1 input for
the metadata diagnostics. Never commit or rewrite the large raw ZIP.

### Observe current metadata conditions for 816 missing source IDs

New `scripts/95_b3_probe_raw_only_replays.py` verifies both frozen
and raw archive SHA-256, recomputes the complete raw/frozen ID
census, and temporarily CRC-extracts source replay members one at
a time. It uses current `sc2reader` to inspect duration, player
count and winner. It samples matched frozen replay identities as
**positive controls** to distinguish discriminatory metadata
conditions from systematic metadata-library limitations.

The first smoke inspects 30 of the 816 unmatched and 15 positive
controls (seed 42), with a new output folder:

```bash
"$SC2_PYTHON" -m unittest discover -s tests \
  -p 'test_b3_raw_only_metadata_diagnostic.py' -v
"$SC2_PYTHON" scripts/95_b3_probe_raw_only_replays.py \
  --archive /home/jlpfritas/HPC-POMDP/ML-SC2/replays.zip \
  --frozen-v31-zip /home/jlpfritas/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean/data/processed/starcraft_full_dataset_v3_1_fixed.zip \
  --max-raw-only 30 --controls 15 --seed 42 \
  --output-dir results/b3_raw_only_metadata_smoke30_20261123
cat results/b3_raw_only_metadata_smoke30_20261123/summary.json
```

If the control cases are valid and current metadata is interpretable,
the full diagnostic can inspect all 816 unmatched replay members,
still one-at-a-time and without persistent extraction, using
`--max-raw-only 816 --controls 100` and a **new output folder**.

An observed `duration_below_min`, `duration_above_max`,
`not_one_vs_one`, `winner_not_single_known_player`, or load failure
is a *contemporary observation*. It does not establish which
condition caused the historical builder to omit that replay.
