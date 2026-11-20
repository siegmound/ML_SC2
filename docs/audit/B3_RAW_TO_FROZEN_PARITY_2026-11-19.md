# B3 — Historical raw replay → frozen v3.1 numeric parity, sample stage

**Date:** 2026-11-19  
**Branch:** audit/sc2-corrections-2026-11-17  
**Status:** NEW READ-ONLY AUDITOR IMPLEMENTED; REAL FAIR RUN NOT YET OBSERVED.

## Why another audit is required

The prior B3 full-table arithmetic audit verified more than 65 million
relationships **inside processed CSVs**. It cannot establish that a
worker count, an army inventory, a scouting score, or a winner label
actually came from the original replay events.

The 300-file raw subset was inventoried on FAIR: 291 filename IDs overlap
the frozen table; 4 other real MPQ replay headers are out of the historical
180–1800-second duration band, while 5 entries are AppleDouble macOS metadata
(194 bytes), not replay payloads.

B3 still requires independently recreating actual features from source
replay bytes. Running today's 38-field clean-room bridge would be a
different algorithm; do **not** treat its mismatches as original-data errors.

## Preserved historical source

The user-provided archive sc2_project_freeze_final.zip contains:
- sc2_parser_v3_counter120.py
- sc2_bridge_v3_counter120.py
- sc2_constants.py
- sc2_counters.py

Two separately uploaded freeze archives contain matching copies. These
historical source *copies* are useful for reconstruction, but not an
attested original execution trace. The archived historical *builder*
has malformed string literals, and must not be silently repaired or
claimed byte-equivalent to whatever was originally executed.

The exact four historical file SHA-256 values pinned in
scripts/79_compare_b3_historical_raw_parity.py are:

| Source | SHA-256 |
| --- | --- |
| Parser | 80ed2d83835c85fc03b4707c5431a2a29aeeb4166fc35777d3ca1bd590b3da1c |
| Bridge | 89886649bfffba88a938f86518dd395ecc7caf0731c10520e3f96a52d12086e4 |
| Constants | 1e568db34d1cbf8ec8ad4f6a7d8fbc8889c90d939c3289b077a74cd5232bb7d0 |
| Counters | ab4fad87512363d096f89d3e305d9c970aa9c0b13354eacb7d39ae64e658ce7b |

A standalone 12 KB ZIP with these four exact script bytes is provided
separately to the operator as
sc2_b3_historical_v31_sources.zip, SHA-256:
4d8888ddace6c209453f8bfd149c16be1e8b69e5e5565d153bafd68df3f60680.

## Experimental comparison

The new scripts/79_compare_b3_historical_raw_parity.py:

1. Rejects changes to the frozen v3.1 ZIP SHA-256.
2. Selects a seeded sample among the **291 inventory-verified** raw IDs
   present in the frozen CSV. Duplicate/ambiguous IDs are not sampled.
3. Hashes each chosen original raw .SC2Replay file.
4. In a temporary directory, materializes the four historical scripts
   **only after checking their exact SHA-256**.
5. Invokes the historical bridge/parser on those replay bytes, then
   retains snapshots where time_sec > 120, as in the historical builder.
6. Streams the original complete frozen ZIP in chunks and keeps only
   rows belonging to sampled replay IDs.
7. Independently checks replay/time membership, winner labels, feature
   names, all matching primitive/derived feature values, and reports
   every discrepancy counted by replay and feature.
8. Writes a new isolated output directory, refuses overwrite, and never
   retrains models or modifies frozen datasets.

It produces summary.json, per_replay.csv, per_feature.csv and
mismatch_examples.csv. A result of SAMPLED_PARITY is reported **only
when** all timestamps, replay labels, 64 feature columns and tested
numeric values agree within rtol=1e-6 / atol=1e-6. A mismatch is
preserved for investigation, not corrected automatically.

This is a **sampled reconstruction using preserved source copies**, not
certification of the original actual generator/version, and not a
proof of raw-to-v3.2 parity.

## Run on FAIR

The original archive or the small extracted source-only ZIP must be
present on FAIR. It is deliberately *not* committed as a new original
build artifact and cannot be substituted with an invented parser.

From the working checkout:

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean
git pull --ff-only origin audit/sc2-corrections-2026-11-17

# Resolve Git LFS post-merge hook PATH warning, if present:
export PATH="$HOME/HPC-POMDP/.tools/git-lfs/bin:$PATH"
git lfs version

source ~/.venvs/ml-sc2/bin/activate
export SC2_PYTHON="$HOME/.venvs/ml-sc2/bin/python"
export PYTHONPATH="$PWD/src"

"$SC2_PYTHON" -m unittest discover \
  -s tests -p 'test_b3_historical_raw_parity.py' -v
```

Transfer the small source-only ZIP to the FAIR checkout, or use an
already available copy of sc2_project_freeze_final.zip. Avoid placing
it inside data/raw/replay_subsets because that directory is the
read-only raw replay source tree.

```bash
"$SC2_PYTHON" scripts/79_compare_b3_historical_raw_parity.py \
  --historical-source-zip sc2_b3_historical_v31_sources.zip \
  --raw-root data/raw/replay_subsets \
  --inventory-csv results/b3_source_inventory_subset_20261119/identity_reconciliation.csv \
  --processed-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip \
  --max-replays 3 --seed 42 \
  --output-dir results/b3_raw_to_v31_parity_sample3_20261119

cat results/b3_raw_to_v31_parity_sample3_20261119/summary.json
cat results/b3_raw_to_v31_parity_sample3_20261119/per_replay.csv
```

**Do not remove or overwrite this output** on re-execution: choose a
distinct sample or output namespace. If the first sample shows
reconstruction/label/feature mismatches, investigate the concrete rows
before expanding sample size. Later a larger seed-stratified sample can
be justified if reproducibility is established.

## B3 sign-off constraints

- The documented 24,057 upstream count is still not a per-file raw
  manifest. 816 remains an unresolved arithmetic difference for the
  entire original corpus, even though the nine out-of-sample entries
  in the small subset have technical explanations.
- A historical script copy matching archived formulas cannot alone
  establish the exact original executed code.
- The nine extra v3.2 combatfix upstream quantities have not been
  recovered and reproduced from actual replay events.
- B3 release flag stays false and B5 stays blocked. Existing B2
  completed evidence and all frozen assets remain immutable.


## Actual FAIR seed-42 sample-3 result: FAILED NUMERIC PARITY

An operator executed the pinned-source historical replay reconstruction on
FAIR with the source-only bundle SHA-256
`4d8888ddace6c209453f8bfd149c16be1e8b69e5e5565d153bafd68df3f60680`
and frozen v3.1 SHA-256
`65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517`.

The four B3 auditor tests passed; the subsequent real sample did NOT
reproduce the numeric features. The interpreter was Python 3.12.3,
NumPy 2.5.5, pandas 3.1.1, sc2reader 1.9.0.

| Replay ID prefix | Frozen rows | Reconstructed rows | Matched timestamps | Label differences | Feature comparisons | Value mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `00bb7e14` | 53 | 53 | 53 | 0 | 3,392 | 786 |
| `0a390370` | 36 | 36 | 36 | 0 | 2,304 | 506 |
| `0b196edc` | 28 | 28 | 28 | 0 | 1,792 | 340 |
| **TOTAL** | **117** | **117** | **117** | **0** | **7,488** | **1,632** |

All three historical outputs had the frozen 64-feature schema, with
zero extra or missing columns. All 117 replay/time keys and winner labels
were reproduced; 5,856/7,488 numerical feature comparisons were within
`rtol=1e-6, atol=1e-6`, but **no full replay passed**.

This is a **real negative replication result**, not an error to suppress
by selecting another tolerance, re-fitting models or regenerating frozen
data. In particular, previous 65.7-million internal algebraic
comparisons can remain correct while primitive replay-event features
or historical execution differ.

**Forensic blocker:** the operator has not yet supplied the existing
`per_feature.csv` and `mismatch_examples.csv`, so the affected
feature families and numerical signs/magnitudes are still unknown.
Inspect these two files before considering a source or environment
change. Do not assert that sc2reader 1.9.0 is incompatible without
feature-level and controlled environment evidence.

The original exact executed parser/bridge version and full historical
replay corpus remain unproven. B3 stays **OPEN / numerical parity FAIL**
and the B5 raw lineage release condition stays false.


## Feature-level diagnosis: concrete historical-source contradiction

The operator aggregated all 7,488 matched-row feature comparisons from
`per_feature.csv`. **46 of the 64 feature columns have zero mismatches**
across all 117 snapshots. The other **18** columns account for every
one of the 1,632 numeric mismatches.

| Family | Fields with mismatches | Mismatches |
| --- | ---: | ---: |
| Primary technology (`p1_tech_level`, `p2_tech_level`, `tech_diff`) | 3 | **259** |
| Primary combat (`p1_combat`, `p2_combat`, `diff_combat_score`) | 3 | 257 |
| Primary counter advantage (p1, p2, diff) | 3 | 297 |
| Derived combat momentum, deltas, windows and volatility | 6 | 534 |
| Derived counter windows and volatility | 3 | 285 |
| **Total** | **18** | **1,632** |

### Proven root cause for the 259 technology mismatches

The historical `sc2_parser_v3_counter120.py` computes each player's
tech level as the maximum of
`self.UNIT_STATS[u].get('tech_level', 0)` for owned known units.
The exact archived `sc2_constants.py` (SHA-256
`1e568db34d1cbf8ec8ad4f6a7d8fbc8889c90d939c3289b077a74cd5232bb7d0`)
contains **no `tech_level` key in any `UNIT_STATS` entry**. Therefore
the recovered source bundle must reconstruct all player tech levels as
zero and `tech_diff = 0`. This follows from source logic, irrespective
of `sc2reader` version or numerical tolerance.

The *actual* frozen v3.1 CSV ZIP
(SHA-256 `65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517`)
was independently inspected for the same three exact replay IDs.
Nonzero frozen technology counts are:

| Field | Frozen nonzero rows (among 117) | Comparison mismatches |
| --- | ---: | ---: |
| `p1_tech_level` | **101** | **101** |
| `p2_tech_level` | **95** | **95** |
| `tech_diff` | **63** | **63** |

Values of `p1_tech_level` and `p2_tech_level` in the frozen rows
include levels **1, 2 and 3**; in the archived-source reconstruction
they are necessarily all **0**. Every mismatch in this three-field
family is therefore **fully explained by the missing `tech_level`
definitions** (259/1,632 = 15.9%).

This is **positive evidence that the archived four-file source combination
cannot be the complete version of the generator that produced frozen v3.1**.
It is *not* evidence that the frozen data itself is wrong, and it cannot
identify the exact original constants revision by itself.

### Remaining 1,373 mismatches — not yet root-caused

The mismatch count for the combat/counter families is **1,373**
(257+297 primary; 534+285 dependent). The recorded examples include
large deviations (e.g. `combat_momentum_15` frozen 0.755973 versus
reconstructed 0.610000 at 705 s), not only micro-scale rounding.

The archived `UNIT_STATS` also lacks the `upg_cat` metadata that
the parser reads for generic weapon/armor upgrade multipliers, so the
saved source combination may omit some upgrade effects. Other plausible
but **unproved** explanations include different `power` values,
`UNIT_COUNTERS` versions or exact per-unit inventories. Matching
worker, income, scouting, unit-type-count, entropy and supply features
makes wholesale time alignment errors less likely but **does not prove**
identical unit-by-unit inventories.

Do **not** repair `UNIT_STATS` with guessed levels/categories and
label the adjusted build historically faithful. Recover a verified
earlier constants/counter revision or compare isolated raw inventories,
upgrade names and counter contributions to frozen values. The feature
formula algebra test does not substitute for this provenance.

### Reproducible static evidence extractor

`scripts/80_audit_b3_source_constants.py` compares an AST-extracted
historical `UNIT_STATS` against the frozen tech values for the exact
replay IDs in `per_replay.csv`, and groups observed per-feature
mismatch totals. It checks historical/frozen input SHA-256, writes
only a new isolated report and refuses overwrite.

```bash
git pull --ff-only origin audit/sc2-corrections-2026-11-17
"$SC2_PYTHON" -m unittest discover -s tests -p 'test_b3_source_constants.py' -v
"$SC2_PYTHON" scripts/80_audit_b3_source_constants.py \
  --historical-source-zip sc2_b3_historical_v31_sources.zip \
  --processed-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip \
  --parity-dir results/b3_raw_to_v31_parity_sample3_20261119 \
  --output-dir results/b3_source_constant_incompatibility_20261119
```

A successful audit of this incompatibility is **not** a PASS of
replay-to-feature parity. The B3 gate and overall release remain blocked.

## Controlled next step — primitive combat and counter forensics

The original mismatch analysis is now established: 46/64 features
match, 18 disagree. All 259 technology mismatches are fully explained
by the absence of tech_level in the exact historical constants,
whereas 1,373 combat/counter and downstream temporal mismatches remain
without a demonstrated upstream cause.

The historical parser's combat score depends on:
- per-unit UNIT_STATS.power and known army inventory;
- opponent unit membership in UNIT_COUNTERS for per-unit penalties;
- completed upgrades filtered by UNIT_STATS.upg_cat (absent in the archive),
  plus per-unit named UNIT_TECH_MULTIPLIERS.

The archived counter score depends on the per-unit UNIT_STATS.power,
known army composition, and the directional UNIT_COUNTERS table. These
dependencies overlap with combat and are not independent evidence.
Recorded worker/supply/entropy parity does NOT establish identical
per-unit army composition, counter edges or upgrade sets.

The new read-only scripts/81_forensic_b3_combat_counter.py reruns
the **same three seed-42 replays** using the SHA-pinned historical
source bundle, temporarily instruments only the in-memory historical
_create_snapshot call, and captures per-player known army compositions,
raw pre-counter/upgrade combat power, completed upgrade names, and
signed frozen-minus-reconstructed residuals for six *primitive*
features: p1/p2 combat, p1/p2 counter and p1/p2 tech.
All original sources, replay data and frozen CSV archives stay unchanged.
Output is written only into a new report directory.

### FAIR run (one small three-replay experiment, no ML retraining)

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean
git pull --ff-only origin audit/sc2-corrections-2026-11-17
export SC2_PYTHON="$HOME/.venvs/ml-sc2/bin/python"
export PYTHONPATH="$PWD/src"

"$SC2_PYTHON" -m unittest discover -s tests \
  -p 'test_b3_combat_counter_forensics.py' -v

"$SC2_PYTHON" scripts/81_forensic_b3_combat_counter.py \
  --historical-source-zip sc2_b3_historical_v31_sources.zip \
  --raw-root data/raw/replay_subsets \
  --inventory-csv results/b3_source_inventory_subset_20261119/identity_reconciliation.csv \
  --processed-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip \
  --max-replays 3 --seed 42 \
  --output-dir results/b3_combat_counter_forensics_sample3_20261119

cat results/b3_combat_counter_forensics_sample3_20261119/summary.json
```

Inspect the output's per_snapshot_primitive_residual.csv and
raw_player_inventories_and_upgrades.csv. Neither the existence
of generic upgrades nor an apparent counter pattern certifies that
missing metadata is the full historical cause. Do not extrapolate
results to all 291 overlapping replays until these primitive causes
are isolated, and do not substitute guessed source constants.

**B3 lineage and raw-to-frozen parity remain open/failed.**

## Actual FAIR script-81 source-instrumentation result — 2026-11-19

The operator successfully ran `81_forensic_b3_combat_counter.py`
against the same three SHA-256-verified raw replay files and the pinned
historical parser, bridge, constants and counter table. All **3/3**
synthetic test cases passed locally, followed by **117 aligned**
snapshots and **234 per-player raw inventory/upgrade observations**.

| Primitive | Miscompares / 117 | Mean absolute error | Max absolute error |
| --- | ---: | ---: | ---: |
| `p1_combat` | 62 | 0.045210 | 0.529215 |
| `p2_combat` | 96 | 0.053954 | 0.294057 |
| `counter_advantage_p1` | 99 | 0.238630 | 0.825000 |
| `counter_advantage_p2` | 99 | 0.248585 | 0.790000 |
| `p1_tech_level` | 101 | 1.290598 | 3.000000 |
| `p2_tech_level` | 95 | 1.418803 | 3.000000 |

These six primitive columns have **552 mismatches in 702 primitive
feature comparisons**; 196 are individual p1/p2 tech, 158 combat
and 198 counter. The additional `tech_diff` mismatch (63) is included
in the separate original 1,632-feature total, not these six columns.
The global 1,632-feature mismatch total is not changed or added to
the 552 primitive statistic: these are overlapping scopes.

The original archive has 51 `UNIT_STATS` records but no `tech_level`
or `upg_cat` fields. The former already explains the technology mismatch;
the latter may explain some upgrade-related combat discrepancies but
no combat/counter primary root cause has yet been certified.

### Controlled direction-of-counter-relation experiment

Source-level inspection found that `sc2_counters.py` lists counters
**effective against** a keyed unit (e.g. Marine maps to Baneling and
Colossus), while the archived `_calc_counter_score` uses the same list
as if the keyed unit counters the listed enemy units. This semantic
disagreement is a concrete hypothesis, not evidence of what executed
when the frozen dataset was built.

New `scripts/82_test_b3_counter_orientation.py` uses **only the
existing script-81 evidence**, with no replay re-parse. It reads the
exact SHA-pinned historical `UNIT_STATS`, `UNIT_TECH_MULTIPLIERS`,
and `UNIT_COUNTERS`; reconstructs the archived counters/combat first;
then tests an explicit reversal of counter-table lookup for both
numeric features, keeping all other definitions and inputs fixed.

For strict attribution: the script-81 compressed inventory records
only known army units. The archived combat penalty may also depend on
non-army/unknown opponent unit types (e.g. Overseer) not captured in
that CSV. If archived combat cannot be independently reconstructed
from the captured evidence for a row, that row is **excluded** from
the semantic combat-direction comparison and separately counted.
Archived *counter* must reproduce exactly or the run fails closed.

The experiment reports both hypotheses' comparison counts, mismatch
rates and absolute errors, plus the number of observed upgrade-level
snapshots. It **cannot identify an attested original execution** or
retroactively select a generator by test fit. The combined metadata
CI checked the four new unit tests, **4/4 PASS** on GitHub run
`44620605003` (commit `df8518a2`).

**Run on FAIR using the files already created by script 81:**

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean
git pull --ff-only origin audit/sc2-corrections-2026-11-17
export SC2_PYTHON="$HOME/.venvs/ml-sc2/bin/python"
"$SC2_PYTHON" -m unittest discover -s tests \
  -p 'test_b3_counter_orientation.py' -v

"$SC2_PYTHON" scripts/82_test_b3_counter_orientation.py \
  --historical-source-zip sc2_b3_historical_v31_sources.zip \
  --forensic-dir results/b3_combat_counter_forensics_sample3_20261119 \
  --output-dir results/b3_counter_orientation_hypothesis_20261119

cat results/b3_counter_orientation_hypothesis_20261119/summary.json
```

Do not rerun the original 3 replay comparisons or change any original
snapshot values just to explore this semantic hypothesis. B3 and B5
release gates remain open until source-lineage proof is obtained.

## Actual FAIR script-82 counter-orientation results — 2026-11-19

The operator reran the four controlled-hypothesis unit tests (**4/4 PASS**)
and evaluated script 82 on the unchanged three 2026-11-19 replay
identities, 117 snapshots and 234 player-snapshot comparisons.
The as-archived counter formula exactly reproduced the independent
script-81 reconstructed predictions. No combat observations were
excluded for incomplete recorded inventory context in this particular
sample.

| Family | Hypothesis | Comparisons | Mismatches | Mean abs. error | Maximum abs. error |
| --- | --- | ---: | ---: | ---: | ---: |
| Counter | As archived | 234 | 198 | 0.243607 | 0.825000 |
| Counter | Table orientation reversed | 234 | **144** | **0.012510** | **0.090000** |
| Combat | As archived | 234 | **158** | 0.049582 | 0.529215 |
| Combat | Table orientation reversed | 234 | 198 | **0.047726** | **0.390615** |

The counter reversal decreases mean absolute error by approximately
**94.9%** and reduces non-matching counter values by **54/234**.
However, **144/234 (61.5%) still differ** beyond 1e-6 tolerances;
the reversed relation is a strong semantic clue rather than numerical
parity, an original-source revision identification, or release proof.

For combat, reversal reduces average absolute error by only ~3.7%,
while **increasing** mismatch count from 158 to 198. This contradicts
applying one blanket counter-direction change to both algorithms.
The sample includes **34** player-snapshot cases with a generic
level-named upgrade and **124** archived-combat mismatches with no
such upgrade event. Therefore *missing generic upgrades alone cannot
explain every combat discrepancy*; this does not establish the root
cause of any particular non-upgrade mismatch.

The previous 1,632 feature-level mismatches and the 259 proven
technology incompatibilities remain the primary original failed
raw-to-frozen parity evidence. The script-82 direction experiment
does not retroactively change them or certify a different model.

### Next diagnostic: residual support and sign (no further fitting)

`scripts/83_profile_b3_counter_residuals.py` inspects only script-82's
already saved per-player snapshot output and the existing script-81
inventory and source hashes. It groups the **144** remaining counter
mismatches by signed frozen-minus-reversed residual, replay, player,
zero/nonzero predictions and normalized frozen/reversed ratio, then
records the 35 largest discrepancies with the observed army type
composition and upgrades. Combat rows with insufficient captured
evidence stay excluded from numerical comparisons.

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean
git pull --ff-only origin audit/sc2-corrections-2026-11-17
export SC2_PYTHON="$HOME/.venvs/ml-sc2/bin/python"
"$SC2_PYTHON" -m unittest discover -s tests \
  -p 'test_b3_counter_residual_profile.py' -v

"$SC2_PYTHON" scripts/83_profile_b3_counter_residuals.py \
  --orientation-dir results/b3_counter_orientation_hypothesis_20261119 \
  --forensic-dir results/b3_combat_counter_forensics_sample3_20261119 \
  --output-dir results/b3_counter_residual_profile_20261119

cat results/b3_counter_residual_profile_20261119/summary.json
head -n 16 results/b3_counter_residual_profile_20261119/top_counter_residuals.csv
```

**No replay reprocessing or model retraining is required** for this
stage. Do not choose new weights or an alternative generator solely
by minimizing errors on the already inspected original test replays.
B3 and B5 release approval remain explicitly blocked.

## Actual FAIR script-83 residual-support profile — 2026-11-19

The operator ran the source-evidence-only residual profiler after the
script-82 reverse-counter experiment. The source JSON has 468 total
hypothesis rows (234 per player snapshot for combat and 234 for
counter), no retraining or new formula selection, and reports:

| Finding | Actual result |
| --- | ---: |
| Counter mismatches after reversing the relation | **144 / 234** |
| Counter absolute-error mean | 0.0125096312 |
| Mean signed frozen-minus-reversed residual | **−0.0112855893** |
| Frozen value less than reversed for mismatches | **122 / 144** |
| Frozen value greater than reversed for mismatches | 22 / 144 |
| Mismatches with both values nonzero and same sign | **144 / 144** |
| Median ratio frozen / reversed (mismatch rows only) | 0.97104593 |
| Counter mismatches for p1 | 96 / 117 |
| Counter mismatches for p2 | 48 / 117 |

By replay, the medians of frozen/reversed for **mismatch rows** are
0.907614 (`00bb7e14…`), 0.972308 (`0a390370…`) and
0.979077 (`0b196edc…`). A single uniform multiplier cannot be
assumed across different armies or replays.

The largest displayed residuals recur in Zerg Queen-heavy contexts.
Examples from the actual snapshot CSV:

| Replay / time / player | Known own army | Frozen counter | Reverse counter | Frozen/reverse |
| --- | --- | ---: | ---: | ---: |
| `00bb7e14…`, 210s, p1 | Queen × 2 | 0.3725 | 0.4250 | **0.876470588** |
| `00bb7e14…`, 300s, p1 | Queen × 3 | 0.55875 | 0.63750 | **0.876470588** |
| `00bb7e14…`, 360s, p1 | Queen × 4 | 0.6385714 | 0.7285714 | **0.876470588** |

The archived historical constants have `Queen.power=8`, so script
82's archived unit weight is `0.75 + 0.25*(8/20) = 0.85`.
If the captured *enemy* unit composition and countered fraction are
also the same as the original frozen build, these Queen-only records
have an implied frozen weight of
`0.85*(0.3725/0.425) = 0.745`.

**This effective weight is conditional, not a proven historical
coefficient**. A difference in enemy inventory/fractions could also
produce the same counter ratio. Merely observing Queen on one side
does not establish that the original generator assigned it a
different power or counter coefficient.

### Next controlled test: single-unit-type numerator/denominator decomposition

New `scripts/84_decompose_b3_single_type_counter.py` uses script 81's
captured army inventories on *both* sides and script 82's saved
reversed-counter values. It independently recomputes the fraction of
countered opponent unit counts for **each single-unit-type army**.
Only where the reversed prediction is algebraically reproduced,
it emits the frozen effective unit weight (frozen / recorded
opponent countered fraction) and groups it by own unit, observed
enemy compositions and replay. Multi-type or zero-fraction armies
are explicitly counted but not assigned imaginary weights.

The test is purely descriptive, not an optimization of historical
weights and not a source repair. It cannot prove equal original
unit inventories in the frozen build or recover the true historical
executable. All previous artifacts stay immutable.

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean
git pull --ff-only origin audit/sc2-corrections-2026-11-17
export SC2_PYTHON="$HOME/.venvs/ml-sc2/bin/python"
"$SC2_PYTHON" -m unittest discover -s tests \
  -p 'test_b3_single_type_counter_decomposition.py' -v

"$SC2_PYTHON" scripts/84_decompose_b3_single_type_counter.py \
  --historical-source-zip sc2_b3_historical_v31_sources.zip \
  --orientation-dir results/b3_counter_orientation_hypothesis_20261119 \
  --forensic-dir results/b3_combat_counter_forensics_sample3_20261119 \
  --output-dir results/b3_single_type_counter_decomposition_20261119

cat results/b3_single_type_counter_decomposition_20261119/summary.json
```

**B3 remains open and B5 is not approved**. Do not adjust the frozen
dataset, unit powers, or parser source because a hypothesis reduces
discrepancies on previously inspected replays.

## Actual FAIR script-84 single-type decomposition and Git source recovery — 2026-11-20

The FAIR operator ran all four source-only synthetic tests (4/4 PASS)
and the real script 84 using the unchanged 2026-11-19 frozen-source
evidence. **234** counter player-snapshot rows were inspected:
**200** had multiple army unit types, **6** had zero countered-enemy
fraction, and **28** permitted an effective-weight calculation
from exactly one known own army type and positive countered fraction.
All **28/28** differ from the archived reversed-lookup prediction.

| Known own unit | Comparable contexts | Distinct opponent compositions | Archived weight | Implied frozen weight |
| --- | ---: | ---: | ---: | ---: |
| Queen | 13 | 7 | 0.8500 | **0.7450** (identical) |
| Adept | 3 | 2 | 0.8000 | **0.7900** (identical) |
| Stalker | 12 | 5 | 0.8125 | **0.781–0.795** (varies) |

These are conditional effective weights **assuming that** the
recorded enemy unit-type counts and counter relations are also those
used when generating the frozen dataset. They are not independently
observed generator parameters or evidence that an attempted repair
is valid. The Queen sample comes from correlated snapshots of one
source replay, not 13 independent matches.

The archived counter unit weight is
`0.75 + 0.25 * min(4, power / 20)`.
For any physically nonnegative unit power this expression is at least
**0.75**; the Queen conditional weight **0.745** is below this
minimum. Thus changing only Queen's nonnegative power is mathematically
insufficient to reconcile this particular isolated Queen evidence
*if* the opponent denominator/fraction is held fixed. Meanwhile, the
varying Stalker effective weight indicates that a fixed per-unit
replacement coefficient alone need not explain all contexts.

### Concrete source-history discovery: Git branch v1.2

GitHub history contains a separate actual version of the project
source files on historical branch `v1.2`, at immutable commit:

`bd64a32b305cc7203d592c7fc9015a01f962d6f9`.

| Source file | Pinned v1.2 Git blob SHA-1 |
| --- | --- |
| `scripts/sc2_parser_v3_counter120.py` | `d839291989936b5867d42f427d419e291f6090da` |
| `scripts/sc2_bridge_v3_counter120.py` | `7856a1c74a5c475bf7414c97772b3461693cb663` |
| `scripts/sc2_constants.py` | `0a0a9faf5e5ea07f696084db37e399a5bb06aac3` |
| `scripts/sc2_counters.py` | `e881ee2513bab91ea8eaaa6852a662c66361e46a` |

The v1.2 constants are **materially different** from the source-only
freeze ZIP. The v1.2 `UNIT_STATS` use `_mk(...)` definitions with
explicit `tech_level`, `attack_upg_cats`, `armor_upg_cats` and
`shield_upgrade`. Its parser's `_get_upgrade_multiplier` reads these
categories. Thus this is a genuine historical alternative capable
of generating **nonzero technology levels**; the incomplete archived
constants cannot do so. Both the parser and counter table have
different byte identities from the freeze ZIP.

**Important caveat:** v1.2 `_calc_counter_score` still uses the
direct `UNIT_COUNTERS[own_unit]` lookup, not the reversed relation
that reduced MAE in the previous descriptive experiment. This revision
should therefore not be presumed to reproduce the frozen counter
values, or to be the actual source generator. A later source revision
or original build variant could still be missing.

### Paired test of a real historical revision, not a guessed patch

New `scripts/85_compare_b3_git_v12_source_parity.py` reads these
four files from the immutable Git commit and independently verifies
all four Git blob IDs, then hashes the extracted bytes with SHA-256.
Using exactly the three seed-42 original replay SHA-256 values and
the same frozen v3.1 ZIP as script 79, it reruns the historical v1.2
bridge and compares all snapshot keys, labels and 64 feature values
at the original 1e-6 tolerances. Baseline identity and replay byte
checks prevent cherry-picking different input files. No original
asset or previously saved result is overwritten.

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean
git pull --ff-only origin audit/sc2-corrections-2026-11-17
git fetch --no-tags origin v1.2

export SC2_PYTHON="$HOME/.venvs/ml-sc2/bin/python"
export PYTHONPATH="$PWD/src"
"$SC2_PYTHON" -m unittest discover -s tests \
  -p 'test_b3_git_v12_source_parity.py' -v

"$SC2_PYTHON" scripts/85_compare_b3_git_v12_source_parity.py \
  --raw-root data/raw/replay_subsets \
  --inventory-csv results/b3_source_inventory_subset_20261119/identity_reconciliation.csv \
  --processed-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip \
  --original-parity-dir results/b3_raw_to_v31_parity_sample3_20261119 \
  --max-replays 3 --seed 42 \
  --output-dir results/b3_git_v12_raw_parity_sample3_20261120

cat results/b3_git_v12_raw_parity_sample3_20261120/summary.json
cat results/b3_git_v12_raw_parity_sample3_20261120/per_replay.csv
```

This is **not** another ML experiment, a new model-selection
procedure or evidence that the historical test set was blind.
The 24,057 vs 23,241 upstream identity gap, full raw corpus
reconstruction, original build attestation and v3.2 upstream
feature provenance are still unresolved. **B3 is OPEN and
B5 is BLOCKED_PENDING_EVIDENCE.**
