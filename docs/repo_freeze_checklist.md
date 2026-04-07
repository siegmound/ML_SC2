# Repo Freeze Checklist

## Data pipeline
- [ ] parser smoke test completed
- [ ] parser audit completed
- [ ] replay failure reasons exported
- [ ] official dataset zip built
- [ ] dataset manifest exported
- [ ] feature audit exported

## Evaluation discipline
- [ ] replay-group-aware splits generated
- [ ] no train/test replay overlap
- [ ] all final models use the same split policy
- [ ] model selection does not touch test

## Model layer
- [ ] logistic baseline completed
- [ ] random forest baseline completed
- [ ] xgboost canonical run completed
- [ ] mlp baseline completed
- [ ] pytorch mlp run completed
- [ ] multi-seed aggregation completed

## Secondary studies
- [ ] ablation study completed
- [ ] temporal study completed
- [ ] calibration study completed
- [ ] matchup study completed or marked unavailable
- [ ] replay-level error analysis completed

## Export and audit
- [ ] result collector completed
- [ ] figures/tables exported from saved artifacts
- [ ] reproducibility check passed
- [ ] internal audit check passed or documented
- [ ] paper and repo pipeline aligned
