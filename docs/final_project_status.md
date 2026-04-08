# Final project status

## Stable final conclusions

- Random Forest and XGBoost are effectively tied as the final tabular candidates.
- Random Forest is marginally stronger as a classifier.
- XGBoost is marginally stronger as a probability-oriented model.
- The best Random Forest profile remains `no_counter`.
- The deep challenger is competitive, but does not surpass Random Forest or XGBoost.

## Calibration conclusion

- Random Forest benefits from post-hoc calibration.
- XGBoost is already well calibrated in uncalibrated form.

Therefore:

- classification-oriented recommendation: **RF + `v3_1_fixed` + `no_counter`**
- probability-oriented recommendation: **XGB + `v3_1_fixed` + full**

## Data provenance status

The upstream replay source is explicitly documented in the repository as **SC2ReSet: StarCraft II Esport Replaypack Set** (Zenodo record `14963356`, DOI `10.5281/zenodo.14963356`, version `2.0.0`).

Provenance documents linked from the final repository state:

- `data/raw/replay_sources.md`
- `data/raw/replaypack_inventory.csv`
- `docs/data_provenance.md`
- `data/processed/manifests/upstream_replay_source_manifest.md`

This closes the main public-facing provenance gap between replay origin, local raw material, and final processed datasets.
