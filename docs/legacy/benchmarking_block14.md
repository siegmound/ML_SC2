# Block 14

Goal: confirm with multi-seed RF profile tests whether `v3_2_combatfix` and the `no_counter` profile remain stronger than the baseline `full` profile.

Compact evaluation regime used for comparability:
- model: RandomForestClassifier
- profiles: `full`, `no_counter`, `no_counter_no_losses`
- n_estimators=100
- max_depth=16
- min_samples_split=5
- min_samples_leaf=2
- max_features=sqrt
- class_weight=None
- max_train_rows=10000
- max_val_rows=4000
- max_test_rows=5000
- seeds: 42, 43, 44

Outputs:
- `results/block14_report/*`
- dataset-specific multi-seed profile tables
- cross-dataset delta tables
