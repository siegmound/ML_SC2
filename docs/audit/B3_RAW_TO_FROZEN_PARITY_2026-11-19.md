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
