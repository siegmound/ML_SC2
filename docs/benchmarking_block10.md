# Block 10 Benchmarking Notes

Block 10 moves beyond the smaller smoke1500 regime and freezes a first **candidate-final** benchmark on a larger subset derived from the full `v3_1_fixed` dataset.

## What was completed

- Materialized `real_v3_1_fixed_smoke3000` from the full zipped dataset.
- Generated replay-group splits for seeds 42, 43, 44.
- Completed a multi-seed canonical-style Logistic Regression benchmark.

## What did not close reliably in this session

- Random Forest candidate-final run on smoke3000.
- XGBoost candidate-final run on smoke3000.

Those models were attempted but were not stable enough in this environment to be marked as completed benchmark outputs.

## Why this still matters

This block establishes the first larger-than-smoke1500 benchmark that is still grounded in the remake pipeline and group-aware replay splitting. It therefore serves as a more credible staging point for candidate-final evaluation than the earlier smoke-only runs.
