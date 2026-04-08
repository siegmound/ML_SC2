# Final project status

## Stable final conclusions

- RF and XGB are effectively tied as final tabular candidates.
- RF is marginally stronger as a classifier.
- XGB is marginally stronger as a probability-oriented model.
- The best RF profile remains `no_counter`.
- The deep challenger is competitive, but does not surpass RF or XGB.

## Calibration conclusion

- RF benefits from post-hoc calibration.
- XGB is already well calibrated in uncalibrated form.

Therefore:

- classification-oriented recommendation: **RF + `v3_1_fixed` + `no_counter`**
- probability-oriented recommendation: **XGB + `v3_1_fixed` + full**

## Data provenance status

The upstream replay source is now explicitly documented in the repository:

- `data/raw/replay_sources.md`
- `data/raw/replaypack_inventory.csv`
- `docs/data_provenance.md`
- `data/processed/manifests/upstream_replay_source_manifest.md`

This closes the main public-facing provenance gap between replay origin, local raw material, and final processed datasets.
