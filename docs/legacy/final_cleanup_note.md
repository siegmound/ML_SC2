# Final cleanup note

This delivery was cleaned before final packaging:

- `delivery/README_delivery.md` was aligned with the actual result directories.
- Superseded XGB runner variants were removed; the canonical full-data XGB runner is `scripts/53_run_xgb_full_gpu_fixed_v3.py`.
- The canonical deep runner is `scripts/64_run_deep_finalist.py`.
- Nested zip snapshots inside `results/` were removed to avoid redundancy and confusion.
- The main `README.md` now includes explicit external dataset names and final recommended entry points.
