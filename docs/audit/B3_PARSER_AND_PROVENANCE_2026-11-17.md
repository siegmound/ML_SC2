# B3 — Parser, feature lineage, and replay provenance

**Date:** 2026-11-17  
**Repository:** `siegmound/ML_SC2`  
**Correction branch:** `audit/sc2-corrections-2026-11-17`  
**Immutable B0 baseline:** `33660e2b4fc4bd75df0824bbb7fa7c2096d13831`

## Summary and sign-off

**B3 code corrections, schema inventory and processed-v3.1 replay census: PASS.**

**Historical replay-to-final-dataset reproduction: OPEN / NOT CERTIFIED.**

This audit explicitly distinguishes corrections to the future clean-room parser
from evidence for the **already frozen datasets**. No historical source ZIP,
train/validation/test split, prediction, training output or model metric was
rewritten in B3.

## A. Verified parser bugs and changes

### Unknown-winner and invalid-match labels

Before B3, `SnapshotParser.build_snapshot` encoded
`winner_pid is None` as `p1_wins=0`, a false negative target.

After B3:
- The replay bridge returns `filtered / missing_winner` for no winner.
- It returns `filtered / ambiguous_winner` if multiple winners are reported.
- It returns `filtered / invalid_winner_pid` if the winner ID does not match
  the two participants.
- Non-1v1 matches are filtered with `not_one_vs_one`; invalid player IDs fail.
- Direct snapshot construction rejects unknown or invalid winner IDs.
- Filtered metadata now reports `parse_status=filtered` consistently.

### `recent_losses` for units born after the starting workers

Before B3, the bridge's `tag_to_owner` pre-scan stopped as soon as
starting positions for two players were found. A unit produced later in a game
could die without an owner mapping and disappear from `recent_losses`.

After B3, the same chronological live-unit-tag inventory used for snapshot
inventory accounting also returns the per-player number of deaths in the
preceding snapshot window, `(t-15,t]` under the standard interval. This
fixes attribution for later-born units in the supported born/init/death event
stream. The test fixture includes player 1 and player 2 units created after
the initial bases; their deaths register at 150 and 165 seconds, respectively.

**Caveat:** Unit morphs, tag reuse, ownership changes and replay-version
differences have not yet been exhaustively validated against real SC2 events.
Correctness for every replay type is not claimed from synthetic tests.

## B. Dataset builder guardrails

The clean-room `DatasetBuilder` now:
- rejects new output filenames equal to the two frozen final dataset ZIPs;
- rejects any existing output ZIP, manifest or audit CSV before writing;
- rejects colliding replay basenames from distinct replaypack paths, because
  the existing bridge uses `Path.name` as its replay identity;
- records source path and explicit failure reasons in new raw-parser audits;
- validates binary winner labels, replay/time uniqueness and constant label
  within each replay;
- records true retained IDs and `all_rows_removed_by_quality_filter`
  exclusions if quality filtering eliminates a replay;
- marks new outputs `independent_clean_room_not_reproduction_of_frozen_v3_1_or_v3_2`.

The builder is intentionally **not** retrofitted to output archived v3.1/v3.2
feature formulas that have not been recovered.

## C. Exact feature *schema* gap

The checker `scripts/68_verify_feature_lineage.py` reads the current parser
snapshot dictionary using Python AST and the saved full XGBoost feature
configurations for seeds 42, 43 and 44 of each dataset variant.

Saved machine-readable file:
`data/processed/manifests/b3_feature_lineage_v1.json`

| Property | Count |
|---|---:|
| Clean-room parser predictors | 38 |
| Final v3.1 predictors | 64 |
| Final v3.2 predictors | 73 |
| Field names shared by clean-room parser and v3.1 | 29 |
| Final v3.1 fields absent from clean-room parser | 35 |
| Clean-room-only fields absent from v3.1 | 9 |
| v3.2 fields added beyond v3.1 | 9 |

The **nine v3.2 additions** are:
`counter_advantage_momentum_15`, `diff_matchup_power`,
`diff_raw_army_power`, `diff_raw_army_power_60s`,
`diff_raw_army_power_120s`, `diff_raw_army_power_vol_120`,
`p1_raw_army_power`, `p2_raw_army_power`, and
`raw_power_momentum_15`.

A matching field name is **not proof of numerical parity**.
The current repository has not yet recovered an end-to-end historical
transform that produces all 64/73 feature values from raw replays.
No invented formulas or guessed missing data were added in B3.

The actual v3.1 CSV ZIP header was checked against the 64-feature ordered
model configuration. The local v3.2 ZIP in this B3 session has an incomplete
central directory and did **not** permit direct header verification;
the v3.2 schema is drawn from three saved full-feature model configs on GitHub.
Re-download and verify the v3.2 LFS payload in B4.

## D. Frozen v3.1 replay census

The streaming checker `scripts/69_audit_replay_lineage.py` read
the original v3.1 frozen archive, verifying every row's
`replay_id`, `time_sec`, and `p1_wins`.

Saved evidence:
`data/processed/manifests/b3_replay_census_v3_1_fixed.json`

| Item | Value |
|---|---:|
| Snapshot rows read | 892,047 |
| Distinct replay IDs | 23,241 |
| Replays with player 1 as winner | 12,122 |
| Snapshot seconds (min–max) | 135–1,785 |
| Duplicated replay/time keys | 0 |
| Conflicting winner labels within replay | 0 |
| Documented upstream replay total | 24,057 |
| Arithmetic difference | 816 |

The figure **24,057** is from the documented source narrative, not from a
complete per-file raw replay inventory in this B3 run. There are **no verified
per-replay reasons for the 816 difference**. The status is explicitly
`UNVERIFIED_NO_RAW_AUDIT`, not `reconciled`.

The new checker can cross-reference a genuine raw-parser CSV with
`--raw-audit-csv`; it rejects missing reasons, duplicate input replay IDs,
or accepted raw IDs that disappear downstream without a filtering log.

## E. Observed tests and exact GitHub code identity

The local B3 fixture was assembled from the original final delivery source,
and every changed source/test file was published to the GitHub correction
branch and compared using Git blob SHA-1 against the locally tested bytes.

- Bridge: `c322da3c7e71d85c1996d0cb2bad295ce0fafea6`
- Snapshot parser: `73bab8d4644e4e2fdbe10ca2ad5256a522034381`
- Dataset builder: `ada615f5c4744045fcf53ec6280d6e1c43722808`
- Feature lineage checker: `432a224ec91a10f2a2b832c3113dc140eb6f631c`
- Replay lineage checker: `d8116f366f23d86e0879bace32d4e4a3f69d6dfd`
- Feature lineage manifest: `eec9c940221197ab425569e601dbd3ad9ebdb2a3`
- v3.1 replay census: `efbf3f20b6094998c6436c0cf7a57d770547a624`

**Test run:** `python -m unittest discover -s tests -p 'test_*.py' -q`  
**Observed result:** **23 tests run, 23 PASS** (B1 + B2 + B3).

Standalone B3 checks:
```bash
python scripts/68_verify_feature_lineage.py
python scripts/68_verify_feature_lineage.py --zip-v3-1 data/processed/starcraft_full_dataset_v3_1_fixed.zip
python scripts/69_audit_replay_lineage.py \
  --processed-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip
```

**Limitations:** the 23/23 suite passed in a local reconstructed test fixture,
not a freshly cloned repository; clean-clone, full Git LFS fetch and real
raw-replay end-to-end comparisons remain B4 and/or additional provenance work.

## F. Remaining gates

- [x] Reject unknown/invalid winner labels and non-1v1 inputs.
- [x] Correct attribution of recent losses after early base detection.
- [x] Protect frozen outputs and detect replay ID basename collisions.
- [x] Preserve exclusion reason codes on new clean-room builds.
- [x] Establish an exact, drift-checkable 38/64/73 feature schema inventory.
- [x] Run frozen v3.1 replay census and validate labels and timestamps.
- [x] Pass local combined regressions (23/23).
- [ ] Recover raw replaypack-level identifiers and the 816 missing replay reasons.
- [ ] Recover and validate all original feature engineering transforms.
- [ ] Verify and count the complete v3.2 LFS ZIP from a fresh checkout.
- [ ] Demonstrate old-vs-new feature value parity on representative raw replays.
- [ ] Pass full clean-clone and dependency audit in B4.

**B3 engineering corrections are complete. The separate historical replay-to-
dataset reconstruction and source exclusion audit are not yet certifiable.**


## Final remote comparison (2026-11-17)

- GitHub reports `main` as **identical** to original B0 commit
  `33660e2b4fc4bd75df0824bbb7fa7c2096d13831`
  (ahead=0, behind=0).
- At the comparison preceding this sign-off, the correction branch was
  **55 commits ahead** over 38 changed paths in total across B0–B3.
- No frozen dataset ZIP, historical RF/XGB/Deep result, compiled PDF or
  PowerPoint appeared in the changed-file list.
- All seven critical B3 local code/schema files were read back from
  GitHub with the same Git blob SHA-1 as the exact locally tested versions;
  `tests/test_b3_feature_lineage.py` also matched.
- The combined B1+B2+B3 suite was executed locally after these changes
  and passed **23/23** tests.
- This final sign-off is itself a documentation-only commit. A full
  clean-clone strict audit and raw replay reconstruction are still pending.

**B3 STATUS: engineering changes verified; historical raw-to-dataset parity
and exclusion provenance unresolved and explicitly not certified.**


## 2026-11-19 addendum — Historical parser source recovered

**This new evidence supersedes only the earlier statement that all 35
non-clean-room v3.1 transformation definitions were unavailable.** The
historical source supplied in two independently matching project
freeze archives, `sc2_parser_v3_counter120.py`, defines the exact
**64 v3.1 predictor names and ordered 67-column final header**. Its
SHA-256 and sampled numeric validation are recorded in
[the source-recovery report](B3_HISTORICAL_V31_SOURCE_RECOVERY_2026-11-19.md).

A streaming check of 15,000 frozen v3.1 snapshots from 393 replay IDs
performed **511,034 independent algebraic comparisons, zero mismatches**
(when the necessary prior snapshot was available). This is meaningful
positive evidence for historical feature formulas, **not a raw replay
extraction reproduction**. The clean-room parser still emits only 38
fields, 816 upstream replays remain without exclusion reasons, and the
nine v3.2-specific feature formulas have not been recovered.

The B3 raw provenance release gate remains **OPEN**, and no frozen
dataset or historical metrics were changed.

## New B3 original source inventory stage — 2026-11-19

The authoritative source reference already tracked in `data/raw/` is
SC2ReSet Zenodo record `14963356` / DOI
`10.5281/zenodo.14963356`. The upstream replaypack catalogue contains
**71 named archive entries**, but the catalogue is not proof those
packages were downloaded or used in the original frozen build.

The user-provided historical freeze archives were reexamined and exact
SHA-256 source identities documented in
[B3_RAW_SOURCE_INVENTORY_2026-11-19.md](B3_RAW_SOURCE_INVENTORY_2026-11-19.md).
No original `.SC2Replay` members were found directly in the uploaded
top-level ZIP directories. The original builder uses a flat `replays/`
directory and does not record individually actionable exclusion reasons.

New `scripts/77_inventory_b3_raw_sources.py` performs a **read-only**
central-directory inventory of replaypacks or loose `.SC2Replay` files,
checks the 71-pack catalogue, independently enumerates frozen replay
IDs, identifies duplicate source basenames and unreconciled identities,
and **never invents failure reasons**. Its synthetic integration tests
(`tests/test_b3_raw_source_inventory.py`) passed **3/3** in the public
GitHub B4 metadata CI on commit `848388ca`, run
`44600118254`. No real raw pack scan was asserted by this CI check.

B3 historical raw-to-feature/exclusion provenance is **STILL OPEN**.
Next evidence depends on access to actual original replaypack ZIPs (on
FAIR, the original workstation, or reacquired from the pinned Zenodo
source). Until then, 816 remains a documented arithmetic difference,
not an individually explained set of parser exclusions. The B5 release
gate remains false.


### Real raw subset available in the tracked repository

After the source-inventory tool was added, an inspection of the tracked
`data/raw/replay_subsets/` directory identified **300 real
`.SC2Replay` files** (plus a placeholder) already available to the
FAIR checkout. The first targeted B3 run should inventory those files
against the frozen 23,241 replay IDs with:
`scripts/77_inventory_b3_raw_sources.py --raw-root data/raw/replay_subsets`.
This is a **partial subset** rather than proof of the original 71-pack
corpus or 24,057 upstream replay count. Details and exact commands:
[B3_RAW_SOURCE_INVENTORY_2026-11-19.md](B3_RAW_SOURCE_INVENTORY_2026-11-19.md).

The B3 original-exclusion/feature-parity gate remains **false**.

## Historical replay-to-feature parity stage prepared — 2026-11-19

The 9 unmatched entries in the tracked 300-file source subset have
now been diagnosed: 4 actual replay headers have durations outside
the original historical builder's 180–1,800-second filter, and 5
are AppleDouble macOS metadata (not MPQ replay payloads). This
is a technical explanation for the **sample**, not reconciliation
of the upstream 24,057-file source selection or original execution.

The next scientific test is **not another derived CSV identity**.
`scripts/79_compare_b3_historical_raw_parity.py` runs an
exact SHA-256-pinned historical v3.1 parser/bridge source copy on
seeded, inventoried raw replay bytes that are present in the frozen
dataset. It streams the original frozen v3.1 data and compares exact
replay/time/label identity and all matching feature values. It reports
per-replay, per-feature and numeric mismatch evidence. The extracted
historical four-file source bundle is supplied separately, not claimed
to be attested original generator execution.

Follow [B3_RAW_TO_FROZEN_PARITY_2026-11-19.md](B3_RAW_TO_FROZEN_PARITY_2026-11-19.md)
for commands and explicit limits. A real FAIR sampled raw-to-frozen
run has **not yet** been observed in this stage. No B3 release flag
can be enabled from source code preparation or the sampled check
alone; full raw source selection and v3.2 combatfix formulas remain open.

## Actual raw-to-feature parity: 3/3 replay FAIL; tech root cause proven

The FAIR reconstructed-source test on seeded 3 matched replay IDs
(117 aligned snapshots, 64 features) returned 7,488 feature comparisons
and 1,632 numeric discrepancies despite all replay/time keys, label
values, and schema names matching. Per-feature investigation found
**46/64 fields exact** and **18 affected fields**, all in the
technology, combat, counter and derived temporal families.

Independent examination of the exact archived `sc2_constants.py`
(51 known UNIT_STATS entries) found zero keys `tech_level` and zero
keys `upg_cat`; the archived parser defaults absent `tech_level` to
zero. Frozen v3.1, however, contains 101 nonzero p1 technology rows,
95 nonzero p2 technology rows and 63 nonzero tech-difference rows
in the same 117-snapshot sample. The exact corresponding mismatch
counts are 101 + 95 + 63 = **259/1,632 (15.9%)**, fully explained
by the historical source-copy metadata deficiency, not tolerance.

The remaining **1,373** combat/counter and temporal mismatches are
**not yet attributed**. Missing `upg_cat` is a possible upstream
upgrade-scoring factor, not a proven numeric explanation. Do not
silently 'fix' the archived constants by inventing tech levels or
upgrading model metrics. See [the updated B3 feature-level investigation]
(B3_RAW_TO_FROZEN_PARITY_2026-11-19.md) and read-only
`scripts/80_audit_b3_source_constants.py`.

**Historical raw-to-final parity remains NOT CERTIFIED and B3/B5 release
gates remain blocked.**

### Script-81 primitive forensics observed / script-82 direction test prepared

The same three FAIR raw replays were instrumented using exact archived
source copies, producing 117 matched snapshots and 234 player-inventory
observations. Primitive comparisons had 552 mismatches / 702 values:
158 combat (p1/p2), 198 counter (p1/p2), 196 player technology.
The 196 player tech mismatches and 63 tech_diff mismatches are covered
by the proven historical constants incompatibility.

The remaining two primitive families are being tested in a controlled
source-semantics experiment: `scripts/82_test_b3_counter_orientation.py`
reuses the script-81 CSV evidence to compare as-archived vs reversed
counter-table direction, without modifying replay bytes or frozen CSV.
Results from the real FAIR experiment are **not yet available**;
neither direction may be identified as the actual original executable.
Some combat contexts may be excluded where the existing compressed
inventory omits relevant non-army enemy units. See
[B3_RAW_TO_FROZEN_PARITY_2026-11-19.md](B3_RAW_TO_FROZEN_PARITY_2026-11-19.md).

**B3 raw source provenance remains OPEN / not certified.**

### Actual counter-orientation experiment: important asymmetry

On FAIR, script 82 used the same 3 raw replays/117 snapshots and the
existing script-81 evidence. **4/4 tests PASS**, and both hypotheses
were compared on 234 complete counter and 234 combat observations.
Reversing UNIT_COUNTERS lookup reduces counter absolute-error mean from
0.243607 to **0.012510 (~94.9%)** and counter mismatches from
198 to **144**; the other 144/234 counter values still fail 1e-6
tolerance, so this is not recovered numerical parity.

On combat, reversal slightly reduces MAE 0.049582 -> 0.047726 but
**increases** mismatch count from 158 to 198. Applying the same
direction change to both primitive families is not supported.
These results show a strong counter score convention mismatch, not
which actual original generator created the frozen dataset.

New `scripts/83_profile_b3_counter_residuals.py` profiles the sign,
zero/nonzero support, relative scaling, replay and side of the
144 remaining reversed-counter discrepancies, from the already
recorded CSVs (without another replay parse). Original replay
membership provenance and execution history remain unverified.
See [B3_RAW_TO_FROZEN_PARITY_2026-11-19.md]
(B3_RAW_TO_FROZEN_PARITY_2026-11-19.md) for the actual FAIR
summary, commands and limitations.

**B3 remains OPEN and release remains BLOCKED_PENDING_EVIDENCE.**

### Actual B3 residual sign/Queen follow-up

FAIR `83_profile_b3_counter_residuals.py` observed 144/234
nonmatching reversed-counter values, all nonzero/same sign;
122/144 frozen values are below the reversed reconstruction,
22 above. Mean signed difference is −0.0112855893 and
mean absolute difference is 0.0125096312. Ratio medians for
the three replays are different (0.907614, 0.972308,
0.979077), so a global scaling is unsupported.

Three highest-contrast Queen-only source observations have the
same frozen/reversed ratio 0.876470588. With the archived
Queen power=8 and counter weight=0.85, this implies a
**conditional effective weight 0.745**, *provided that* the
original frozen generator used the opponent unit fractions captured
by the reconstructed replay snapshots. The actual original
counter coefficient is not attested by this equality.

New `84_decompose_b3_single_type_counter.py` checks those
opponent fractions explicitly for every recorded one-unit-type
army without replay parsing or numerical formula fitting,
groups conditional weights by unit and opponent composition,
and rejects previously altered source evidence. Its four
synthetic regression tests passed in B4 metadata CI run
`44625740137`; a real FAIR script-84 execution remains pending.

See [B3_RAW_TO_FROZEN_PARITY_2026-11-19.md]
(B3_RAW_TO_FROZEN_PARITY_2026-11-19.md). The 816 historical
upstream count difference and full replay-to-feature provenance
are still unresolved; **B3 remains OPEN**.
