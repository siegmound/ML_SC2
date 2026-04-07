# Methodology

## Canonical pipeline

1. validate parser behavior on a fixed replay subset
2. audit replay parsing outcomes with reason codes
3. build a canonical dataset with explicit manifest metadata
4. run feature integrity checks before model training
5. generate replay-group-aware splits
6. train baselines and main models using the same split discipline
7. run secondary studies: ablation, temporal, matchup, calibration, error analysis
8. collect and export results into machine-readable and paper-ready forms

## Evaluation discipline

- replay IDs are the grouping key
- no replay may appear across train/validation/test boundaries
- no hyperparameter selection may use the test set
- calibration must be fit on validation-only data
- final claims should be supported by multi-seed summaries when available

## Dataset policy

Heavy datasets stay zipped in the repo. Each official dataset must have:

- a zipped storage artifact
- a manifest JSON
- feature schema information
- inclusion/exclusion metadata

## Design rationale

The remake intentionally prioritizes auditability over convenience. It is better to have fewer features and fewer models with strong manifests than a larger set of scripts that cannot be reproduced cleanly.
