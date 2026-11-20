# B2 — FAIR full-scale retrospective numerical review (2026-11-19)

**Branch:** `audit/sc2-corrections-2026-11-17`  
**Dataset:** frozen `real_v3_1_fixed`, originally 64 features  
**Source of numbers:** operator-provided FAIR stdout from
`scripts/76_summarize_b2_full_fair.py`, reproduced verbatim below at six
decimals; its execution first ran `75_verify_b2_full_fair.py`: **9/9 PASS**.
The two aggregator unit tests also reported **2/2 PASS**.

**Limit:** The nine individual `metrics_summary.json` files, the 21-row
`per_seed_metrics.csv` and the SHA-256 `evidence_manifest.json` are still on
FAIR. Their contents and hashes were **not** independently copied into this
repository for this review. This report records the observed stdout only;
it does not claim a separately retrieved binary/prediction audit.

## 1. Observed verified aggregate table

All values are **arithmetic mean ± sample standard deviation** across seeds
42, 43 and 44 (`ddof=1`, `n=3`). These are descriptive dispersion
statistics, **not** confidence intervals, independent replications or
significance tests. RF and XGB variants below evaluate the same original
historically inspected test partitions; Deep uses the same original test
partitions but a different validation/tuning protocol.

| Model | Variant | Accuracy | Balanced accuracy | ROC-AUC | Log loss ↓ | Brier ↓ | ECE(15) ↓ | MCE(15) ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF | uncalibrated | 0.612187 ± 0.006253 | 0.609963 ± 0.005879 | 0.658023 ± 0.006005 | 0.651719 ± 0.002429 | 0.230221 ± 0.001147 | 0.025277 ± 0.008138 | 0.081053 ± 0.012472 |
| RF | Platt | 0.611196 ± 0.007747 | 0.609789 ± 0.005719 | 0.658023 ± 0.006005 | 0.650959 ± 0.004058 | 0.229929 ± 0.001898 | 0.022741 ± 0.005102 | 0.060708 ± 0.007243 |
| RF | isotonic | 0.610715 ± 0.005342 | 0.609986 ± 0.005704 | 0.657792 ± 0.005948 | 0.652160 ± 0.004585 | 0.230026 ± 0.001827 | 0.022596 ± 0.006206 | 0.068657 ± 0.041327 |
| XGB | uncalibrated | 0.612908 ± 0.006416 | 0.609444 ± 0.005833 | 0.659942 ± 0.006019 | 0.648363 ± 0.003095 | 0.228836 ± 0.001471 | 0.012622 ± 0.007196 | 0.034734 ± 0.006555 |
| XGB | Platt | 0.612151 ± 0.007204 | 0.610210 ± 0.005093 | 0.659942 ± 0.006019 | 0.649502 ± 0.004105 | 0.229293 ± 0.001933 | 0.022360 ± 0.004425 | 0.073876 ± 0.026160 |
| XGB | isotonic | 0.610676 ± 0.004888 | 0.610481 ± 0.005300 | 0.659710 ± 0.005977 | 0.650521 ± 0.005131 | 0.229381 ± 0.001877 | 0.022632 ± 0.004911 | 0.075707 ± 0.037608 |
| Deep MLP | selected candidate only | 0.607606 ± 0.004807 | 0.606861 ± 0.006102 | 0.656501 ± 0.005573 | 0.651373 ± 0.002302 | — | — | — |

Deep's original summary did not save Brier, ECE and MCE; missing entries
are intentionally absent, not estimated as zero.

## 2. Descriptive interpretation; no post-hoc model selection

- **XGB uncalibrated vs RF uncalibrated:** accuracy difference
  `+0.000721` (0.0721 percentage points); ROC-AUC difference
  `+0.001919`; log-loss difference `-0.003356`; ECE difference
  `-0.012655`. These are small descriptive gaps on different feature
  profiles, not evidence of a statistically proved superior algorithm.
- **XGB uncalibrated vs Deep finalist:** accuracy difference
  `+0.005302` (0.5302 percentage points), ROC-AUC `+0.003441`,
  log loss `-0.003010`. The Deep candidate is chosen using the original
  validation groups, while XGB uses separate tuning/calibration groups.
- **RF calibrators:** Platt improves mean log loss by `0.000760`,
  ECE by `0.002536` and MCE by `0.020345` relative to raw RF; its
  mean accuracy falls by `0.000991`. Isotonic slightly reduces ECE but
  worsens mean log loss and accuracy relative to raw RF.
- **XGB calibrators:** both Platt and isotonic have worse mean log loss,
  Brier, ECE and MCE than raw XGB on this retrospectively inspected test
  partition. For Platt, ECE rises from `0.012622` to `0.022360`; for
  isotonic, to `0.022632`. **Do not** interpret this retrospective
  comparison as license to select the calibrated/uncalibrated production
  variant by test score.

All models operate near **61% snapshot accuracy**; there is no basis here
for claiming near-perfect outcome prediction or a large modeling advance.
Because replay snapshots cluster within matches and seeds reuse the overall
dataset, snapshot counts are not independent experimental units, and these
three seed-level means cannot justify a formal significance claim.

## 3. Comparison with historical train-only retrospective evidence

The preexisting `docs/audit/B2_METHODS_2026-11-17.md` reports historical
three-seed means under *different* retrospective methods:

| Metric | Archived RF train-only | Current RF uncalibrated | Archived XGB train-only | Current XGB uncalibrated |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.612270 | 0.612187 | 0.612792 | 0.612908 |
| Balanced accuracy | 0.610068 | 0.609963 | 0.609328 | 0.609444 |
| ROC-AUC | 0.658147 | 0.658023 | 0.660058 | 0.659942 |
| Log loss ↓ | 0.651698 | 0.651719 | 0.648437 | 0.648363 |

The ranking and scale remain similar; these are **not** new independent
test samples. Differences should not be ascribed to one methodological
change without isolating all training-budget, tuning and feature choices.

## 4. Scientific interpretation / B5 publication handoff

1. Continue to describe the result as a **retrospective audit**, not as
   an untouched held-out test benchmark; earlier development saw the tests.
2. Report the corrected strict four-way RF/XGB and selected-only Deep
   numbers as a **separate B2 table**, not an undocumented replacement of
   historical Block 16 or v3.2 headline metrics.
3. Make the three-way Deep vs four-way RF/XGB protocol and the RF
   `no_counter` vs XGB `full` profiles explicit in captions.
4. The new run reports use **sample SD**; the historical main-paper v3.2
   headline table uses **population SD**. Keep those conventions labelled,
   not numerically conflated.
5. Do not select calibrator or finalist by the historical test
   performance. If a calibration choice is needed for future prospective
   use, freeze a selection rule on a separated tuning/calibration set
   before a genuinely untouched future evaluation.
6. No new paper PDF or PowerPoint is authorized by this report; reconcile
   scientific text/figures in B5's distinct review workflow.

## 5. Stage disposition

**B2 full-scale execution and 3-seed descriptive numerical review:
COMPLETED within the stated retrospective scope**, based on the operator
transcript, with no new blind test or formal superiority result.

The input `analysis/` directory remains the canonical **FAIR-generated**
numeric source. For archival-grade traceability, the small generated
`three_seed_summary.csv`, `per_seed_metrics.csv` and
`evidence_manifest.json` should still be copied under version control
without committing the large per-snapshot prediction artifacts.

**Project release:** remains `BLOCKED_PENDING_EVIDENCE` until separate
B3 raw replay provenance, B5 corrected paper/deck review and explicit
release-candidate approval have passed.
