# B3 — Recovered historical v3.1 feature source (partial provenance)

**Recorded:** 2026-11-19  
**Branch:** `audit/sc2-corrections-2026-11-17`  
**Status:** historical v3.1 feature definitions **RECOVERED**; original
raw replay-to-final pipeline and upstream exclusions **NOT CERTIFIED**.

## 1. Source inventory and exact identities

A separate user-supplied archive
`sc2_project_freeze_final_with_pomdp_appendix.zip` includes a historical
replay parser at
`sc2_project_freeze_final/scripts/sc2_parser_v3_counter120.py`.
The same exact source bytes also occur in
`sc2_project_freeze_final.zip`.

- Historical parser SHA-256:
  `80ed2d83835c85fc03b4707c5431a2a29aeeb4166fc35777d3ca1bd590b3da1c`.
- Its `snapshot = {...}` dictionary includes **67 fields**:
  `replay_id`, `time_sec`, 64 predictors and `p1_wins`.
- The order and spelling of all **67 keys match exactly** the CSV header
  in the frozen `starcraft_full_dataset_v3_1_fixed.zip`, not merely a set
  intersection. There are no missing or extra predictors.
- The source also includes explicit formulas for the counter score,
  worker and combat differences, 15-second momenta, 60/120-second trends,
  worker growth, and 120-second rolling standard deviations. The
  accompanying historical replay bridge, constants and counter tables
  have been preserved in the user's separate source-recovery package.

**Interpretation:** the former B3 inventory showed **35 fields absent from
the current 38-feature clean-room parser**. Those 35 fields are not
missing from this *recovered historical parser source*; the 38/64/73
schema comparison remains correct for the current clean-room implementation.
Source-name equality is not proof that this precise historical parser
was the final data-generation executable or that all feature values
match upstream raw events.

## 2. Independent algebraic check of frozen v3.1 data

A read-only streaming numeric check of the **first 15,000 frozen v3.1
snapshot rows**, representing **393 replay IDs**, evaluated 37 classes
of static and historical feature identities using `math.isclose`
with absolute and relative tolerance `1e-6`.

- **511,034 individual formula comparisons**, **0 mismatches**.
- Cases requiring known prior snapshots (15, 60, 120 seconds) were
  checked only when that exact earlier time was present in the sample.
  This avoids fabricating unseen history from before the dataset's
  135-second retention threshold.
- Population rolling standard deviation was evaluated as
  `numpy.std`, as specified in the historical parser.
- A report with per-feature counts and a portable audit checker was
  placed in a separate review package
  `sc2_b3_historical_v31_recovery_2026-11-19.zip`;
  package SHA-256:
  `8df89cbcaaac95649a606d8d04897f3706202aba45e38bba11b9de6cc5d61419`.
  This package is an audit deliverable, **not** a replacement for
  a tracked original replay provenance inventory.

This supplies **stronger derived-feature evidence** than schema-only
comparison; it does not establish numerical parity for all 892,047 rows,
nor raw replay extraction parity.

## 3. Explicitly unresolved parts of B3

1. The documented 24,057 raw replay count and the 23,241 final replay
   IDs still differ by **816 replays** with **no verified individual
   exclusion reason records**.
2. An actual replaypack inventory (or historical parser audit CSV),
   bridge-version provenance and representative replay-to-final numeric
   comparisons are still required for original source-level certification.
3. The historical v3.1 parser does not provide the nine extra v3.2
   `combatfix` feature implementations; archived full v3.2 formulas and
   raw-replay parity must be independently established.
4. Version and code provenance need confirmation: a historical archive
   source that matches the column header and algebra is not an attested
   execution trace of the originally frozen dataset.

No training run, original dataset ZIP, replay split, prediction CSV,
canonical PDF or original PPTX was modified by this investigation.

**B3 release gate remains false** until all raw replay provenance,
downstream exclusions and historical raw-to-final parity are resolved.
