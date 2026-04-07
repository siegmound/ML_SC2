# Block 8 Benchmarking

Block 8 introduces staged training for heavier models and reproducible smoke benchmarks on real zipped datasets.

## Goals

- make RF and XGB practical on smoke subsets
- preserve canonical artifacts and split discipline
- separate smoke-stage search from standard-stage search
- document a staged path toward full dataset runs

## Key additions

- dataset-specific split manifests (`{dataset_name}_split_seed_{seed}.json`)
- row-budgeted smoke training for RF and XGB
- staged XGB search spaces: smoke / standard / extended
- benchmark runner for smoke subsets
- staged run-plan generator for full dataset execution
- legacy-vs-remake summary export

## Recommended execution order

1. materialize smoke subset
2. generate dataset-specific split manifest
3. run `28_run_smoke_benchmarks.py`
4. collect results
5. generate block 8 report
6. prepare staged full-run plan
