# S0 — Sequential learning extension: branch and handoff

**Repository:** `siegmound/ML_SC2`
**Upstream correction branch:** `audit/sc2-corrections-2026-11-17`
**Upstream commit at branch creation:** `397dc3c12a7ae006168a46037881e902cde1c92d`
**Development branch:** `feature/sc2-sequential-models`
**Status:** S0 GitHub setup complete; FAIR workstation validation pending.

## Objective

Implement a new supervised temporal experiment using *unchanged* frozen v3.1 snapshot archives. Evaluate whether sequences of replay snapshots improve classification and probability quality relative to tabular baselines when predictions are compared on exactly the same replay/time endpoints.

The extension is **not** an implementation of the POMDP appendix and is **not** a claim of reconstruction of the missing replay-to-64-feature historical pipeline.

## Guardrails

1. Do not modify `main`, `audit/sc2-corrections-2026-11-17`, historical final metrics, saved predictions, split JSON, frozen ZIPs or B0 protected objects.
2. Read `data/processed/starcraft_full_dataset_v3_1_fixed.zip`, never overwrite it; use `replay_id` as the group, `time_sec` as the ordered timeline and `p1_wins` as the target.
3. Preserve the original seed 42/43/44 train/validation/test replay partitions, and create windows **within each replay and split**.
4. A temporal window ends at time `t` and may contain only the rows at or before `t`. Require consecutive 15-second steps; do not bridge gaps.
5. Fit scalers, imputers and all learned transformations on training only. Never use test predictions to select candidates.
6. Keep sequential outputs under `results/sequential/`, configs under `configs/sequential/`, module code under `src/sc2proj/sequential/` and regression tests under `tests/`.
7. Baselines and sequential models must be evaluated on **identical endpoint keys** (`replay_id`, `time_sec`), including common-endpoint comparisons for different window lengths.
8. State limitations explicitly: historical test labels were already viewed, and historical feature formulas have not all been reproduced end to end.

## Phased implementation

- **S0** (this commit): isolate a development branch, document scope and workstation validation. No model/dataset change.
- **S1**: add read-only sequence window index/builder, feature-schema validation, strict temporal continuity, CLI and unit tests. Initial windows: 9, 17, 33; stride 1. Do not duplicate all sequence tensors on disk.
- **S2**: preserve replay-aware splits, train-only preprocessing, stable feature ordering, memory-safe PyTorch Dataset/DataLoader and leakage tests.
- **S3**: endpoint-matched tabular baselines, alignment audit, metric report.
- **S4**: GRU MVP; validation-based selection; checkpoint and prediction manifests; local smoke test before full-data training.
- **S5**: LSTM and causal Temporal CNN comparators.
- **S6**: multi-seed 9/17/33 window comparison, with same common endpoints, raw-state vs engineered-temporal feature ablations, replay-cluster uncertainty and probability metrics.
- **S7**: reproducibility, final visualizations and documentation with reproducible commands. New results are separate from historical conclusions.

## FAIR workstation — S0 checkout

From a **clean, existing local clone** of `ML_SC2` (save or commit unrelated local changes first):

```bash
git fetch origin
git switch --track origin/feature/sc2-sequential-models
git status --short --branch
git rev-parse HEAD
git merge-base --is-ancestor 397dc3c12a7ae006168a46037881e902cde1c92d HEAD
git diff --exit-code origin/audit/sc2-corrections-2026-11-17 -- data/processed results tables src scripts
python scripts/70_verify_freeze.py --mode pointer
python scripts/58_verify_final_table.py --check
python -m unittest discover -s tests -p 'test_*.py' -q
```

Expected:

- The branch is `feature/sc2-sequential-models`, clean and based on the correction branch.
- The merge-base and diff checks return status 0.
- B0 protected historical blob/LFS pointer audit passes (no large LFS payload download needed in S0).
- The final table check and existing regression suite pass **on FAIR**; this is a gate, not a claimed result.

Please report the actual `git rev-parse HEAD` value and the full output of any failed command. If a command fails due to missing dependencies, report the package error rather than modifying the historical environment blindly.

## Release policy

Each following phase is committed by the assistant to this feature branch, then exercised by the user on FAIR before approval of the next phase. No automatic merge into the correction branch or main, no rewriting historical artifacts, and no unverified full-training claims.
