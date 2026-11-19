# B2 — FAIR execution and verification runbook (2026-11-19)

**Branch:** `audit/sc2-corrections-2026-11-17` only.  
**Scientific status:** NOT YET RUN / NOT CERTIFIED.  
**Do not touch:** `main`, `feature/sc2-sequential-models`, frozen data, historical predictions, B0 artifacts, canonical PDFs/PPTXs.

## Experimental contract

Nine new full-data jobs on the original **v3.1 fixed** data:
- Seeds **42/43/44** × **RF no_counter**: fit train only, strict four-way replay partition, calibration on replay-disjoint calibration portion of original validation; predict uncalibrated/Platt/isotonic.
- Seeds **42/43/44** × **XGBoost full**: fit train only, use tuning subset for early stopping, calibration subset for Platt/isotonic; evaluate on the previously inspected original test replay groups.
- Seeds **42/43/44** × **Deep MLP**: candidate search on original train/validation groups, single selected-candidate test prediction after model selection, GPU required.

Deep uses the original **three-way** train/validation/test split, not the strict four-way RF/XGB calibration split. This is a methodological difference and must not be hidden when comparing results. All test partitions were already inspected historically: do not claim a newly blind holdout.

Default results root: `results/b2_fullscale_audit_20261119` (new and isolated).
Run directory existence is not sufficient for resuming: the launcher calls
`scripts/75_verify_b2_full_fair.py` to re-check all saved prediction rows,
frozen replay groups and metrics before skipping. Incomplete directories are
never overwritten; select a **new** `B2_OUT_BASE` for a fresh attempt.
The run verifier is deliberately read-only.

## 1. Synchronize the existing FAIR clean checkout

```bash
cd "$HOME/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean"
git fetch origin
git switch audit/sc2-corrections-2026-11-17
git pull --ff-only origin audit/sc2-corrections-2026-11-17
git status --short
git log -1 --format='%h %s'
```

Changes were committed through GitHub directly. You **pull** them on FAIR;
do not push large generated training files. If making further local commits:
`git push origin audit/sc2-corrections-2026-11-17` (never force push).

## 2. Select locked FAIR runtime and validate

```bash
export PATH="$HOME/HPC-POMDP/.tools/git-lfs/bin:$PATH"
# FAIR working environment verified in the 2026-11-19 run:
source "$HOME/.venvs/ml-sc2/bin/activate"
export SC2_PYTHON="$(command -v python)"
export PYTHONPATH="$PWD/src"
export B2_THREADS=8
export B2_GPU=0
nvidia-smi
git lfs pull --include="data/processed/starcraft_full_dataset_v3_1_fixed.zip,data/processed/starcraft_full_dataset_v3_2_combatfix.zip" --exclude=""
bash -n scripts/74_run_b2_full_fair.sh
"$SC2_PYTHON" -m compileall -q scripts/64_run_deep_finalist.py scripts/67_run_strict_calibration.py scripts/75_verify_b2_full_fair.py tests/test_b2_full_fair.py
"$SC2_PYTHON" -m unittest discover -s tests -p 'test_b2_methodology.py' -v
"$SC2_PYTHON" -m unittest discover -s tests -p 'test_b2_full_fair.py' -v
bash scripts/74_run_b2_full_fair.sh --plan
bash scripts/74_run_b2_full_fair.sh --preflight
```

The preflight checks exact full dataset Git LFS SHA-256/headers, protected
historical artifacts, original matched train-only evidence, Python 3.12 and
CUDA. This is an integrity check; it does **not** train anything.

## 3. Run one verified full-scale job first

```bash
bash scripts/74_run_b2_full_fair.sh --execute --seed 42 --model rf
bash scripts/74_run_b2_full_fair.sh --verify --seed 42 --model rf
```

This is a genuine full-scale RF seed-42 job, not a lightweight smoke subset.
Inspect `results/b2_fullscale_audit_20261119/logs/rf_seed_42.log` and the
new strict-calibration `metrics_summary.json` before starting all models.

## 4. Complete remaining jobs and verify all nine

Use a terminal multiplexer (such as `tmux`) when connecting over SSH,
otherwise disconnecting may terminate the process.

```bash
tmux new -s sc2-b2-fair
# Inside tmux, from the same checkout with exports above:
bash scripts/74_run_b2_full_fair.sh --execute
bash scripts/74_run_b2_full_fair.sh --verify
```

`--execute` skips the previously completed RF seed 42 **only if the verifier
passes**. It then runs the remaining seed/model combinations. The launcher
writes separate `logs/{rf,xgb,deep}_seed_{42,43,44}.log` files. Use
`tail -f results/b2_fullscale_audit_20261119/logs/xgb_seed_42.log` from
another terminal to inspect progress. In tmux, detach with Ctrl+B then D;
reattach with `tmux attach -t sc2-b2-fair`.

**Important for the tmux example:** environment variables exported before
starting tmux are inherited. If opening a fresh login shell/session, repeat
the exports from section 2, especially `SC2_PYTHON` and `PYTHONPATH`.

If a run is interrupted, do **not** delete or replace partial outputs; keep
logs for forensics and choose an isolated name, e.g.:
```bash
export B2_OUT_BASE=results/b2_fullscale_audit_retry01
bash scripts/74_run_b2_full_fair.sh --plan
```
That starts a new complete experiment namespace; it does not magically
resume estimator training from a partial checkpoint.

## 5. Interpretation and release status

Verify all nine reports, split manifests, feature profiles, saved test
predictions, individual seed variability and calibration deltas. Preserve the
full commands, Git SHA, runtime identity and logs in a review report. Do not
replace historical results/figures before the comparison and paper audit.

B2 remains `OPEN` until the **real** FAIR runs finish and their evidence is
reviewed. The independent B3 raw replay provenance and release approval
remain open even after a successful B2.

## Actual execution outcome (operator transcript, 2026-11-19)

The pre-execution NOT YET RUN label above is historical. All nine runs
completed on FAIR and `--verify` returned **9/9 PASS**. The actual
interpreter was `$HOME/.venvs/ml-sc2/bin/python` (Python 3.12.3,
PyTorch 2.15.0+cu132, XGBoost 3.4.1); the earlier
`ml-sc2-b4-offline` example path did not exist on that workstation.

The saved numeric metrics are in the FAIR result namespace, not embedded
in the terminal transcript. For final descriptive comparison and input
SHA-256 manifests, run:

```bash
"$SC2_PYTHON" -m unittest discover -s tests -p 'test_b2_summary.py' -v
"$SC2_PYTHON" scripts/76_summarize_b2_full_fair.py
```

See [full-size FAIR verification handoff](B2_FULLSCALE_FAIR_VERIFICATION_2026-11-19.md).
The release remains blocked on numeric review, B3 provenance and B5 approval.
