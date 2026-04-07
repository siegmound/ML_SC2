# Final status matrix

## Solid
- Full multi-seed RF vs XGB on `v3_1_fixed`
- Full multi-seed RF vs XGB on `v3_2_combatfix`
- RF best profile = `no_counter`
- Deep challenger tested on full `v3_1_fixed`
- Rigorous calibration on validation->test for the final candidates

## Corrected intermediate narrative
- `v3_2_combatfix` is **not** clearly better than `v3_1_fixed` at final scale
- XGB is **not** a dominant final winner over RF
- counter features are **not** required by the best RF pipeline

## Final recommendations
- Classification-oriented: RF + `v3_1_fixed` + `no_counter`
- Probability-oriented: XGB + `v3_1_fixed` + full features
