# B2 FAIR full-scale — three-seed descriptive results

**Evidence policy:** all nine runs must pass the independent B2 verifier before this report is produced.
**Test status:** original replay-level holdouts had been inspected historically; these are retrospective tests, not pristine blind evaluation.
**Protocols:** RF/XGB use disjoint train/tune/calibration/test groups; Deep uses train/validation/test, selecting one candidate by validation before test inference.
**Uncertainty:** arithmetic mean ± sample standard deviation across seeds 42, 43 and 44 (n=3), not a confidence interval or significance test.
**Calibration:** RF/XGB variants are all reported; no calibration method is selected by its test score. Deep has no saved Brier/ECE/MCE in its run summary, hence those cells remain unavailable.

| Model | Variant | Accuracy | Balanced accuracy | ROC-AUC | Log loss | Brier | ECE (15) | MCE (15) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rf | uncalibrated | 0.612187 ± 0.006253 | 0.609963 ± 0.005879 | 0.658023 ± 0.006005 | 0.651719 ± 0.002429 | 0.230221 ± 0.001147 | 0.025277 ± 0.008138 | 0.081053 ± 0.012472 |
| rf | platt | 0.611196 ± 0.007747 | 0.609789 ± 0.005719 | 0.658023 ± 0.006005 | 0.650959 ± 0.004058 | 0.229929 ± 0.001898 | 0.022741 ± 0.005102 | 0.060708 ± 0.007243 |
| rf | isotonic | 0.610715 ± 0.005342 | 0.609986 ± 0.005704 | 0.657792 ± 0.005948 | 0.652160 ± 0.004585 | 0.230026 ± 0.001827 | 0.022596 ± 0.006206 | 0.068657 ± 0.041327 |
| xgb | uncalibrated | 0.612908 ± 0.006416 | 0.609444 ± 0.005833 | 0.659942 ± 0.006019 | 0.648363 ± 0.003095 | 0.228836 ± 0.001471 | 0.012622 ± 0.007196 | 0.034734 ± 0.006555 |
| xgb | platt | 0.612151 ± 0.007204 | 0.610210 ± 0.005093 | 0.659942 ± 0.006019 | 0.649502 ± 0.004105 | 0.229293 ± 0.001933 | 0.022360 ± 0.004425 | 0.073876 ± 0.026160 |
| xgb | isotonic | 0.610676 ± 0.004888 | 0.610481 ± 0.005300 | 0.659710 ± 0.005977 | 0.650521 ± 0.005131 | 0.229381 ± 0.001877 | 0.022632 ± 0.004911 | 0.075707 ± 0.037608 |
| deep | selected_candidate_only | 0.607606 ± 0.004807 | 0.606861 ± 0.006102 | 0.656501 ± 0.005573 | 0.651373 ± 0.002302 | — | — | — |

Accuracy, balanced accuracy and ROC-AUC: higher is better. Log loss, Brier, ECE and MCE: lower is better.
The RF no_counter and XGB full feature profiles differ. Comparisons are descriptive and cannot establish statistically significant superiority from only three correlated seed splits.
