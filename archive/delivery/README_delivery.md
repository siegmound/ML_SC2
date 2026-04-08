# Final Delivery Package

This package is the final, reproducible project snapshot after:
- full multi-seed RF vs XGB comparison
- full dataset-version comparison (`v3_1_fixed` vs `v3_2_combatfix`)
- rigorous calibration
- final deep challenger
- final report refinement

## Recommended use
- `docs/final_report_refined.md` for the final written summary
- `paper/final_results_refinement.tex` for LaTeX integration
- `tables/` for paper-ready CSV tables
- `results/block16_final/`, `results/final_calibration/`, and the final files in `docs/`, `paper/`, and `tables/` for final structured summaries
- `scripts/64_run_deep_finalist.py` is the cleaned version without the NumPy non-writable tensor warning issue

## Final recommendations
- Classification-oriented: RF + `v3_1_fixed` + `no_counter`
- Probability-oriented: XGB + `v3_1_fixed` + full features
