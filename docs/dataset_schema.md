# Dataset schema

Mandatory columns:
- replay_id
- time_sec
- p1_wins

Feature columns are versioned through dataset manifests.
The canonical pipeline stores one manifest JSON per dataset build.


## Prediction artifact schema

Canonical imported prediction files should contain at least:
- `replay_id`
- `time_sec`
- `y_true`

Recommended additional columns:
- one probability column such as `y_prob`, `y_prob_xgb`, `y_prob_rf`, `y_prob_mlp`
- one hard prediction column such as `y_pred`, `y_pred_xgb`, `y_pred_rf`, `y_pred_mlp`

## Legacy compatibility note

If only frozen result artifacts are available, the remake treats them as `artifact_only` inputs.
In that case it can still:
- inventory the zip contents
- import summaries and prediction CSVs
- build manifests
- validate prediction schemas
- report what is still missing for a full canonical rerun
