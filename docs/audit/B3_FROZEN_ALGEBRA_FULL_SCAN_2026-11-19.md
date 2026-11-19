# B3: Frozen full-table algebra consistency audit — 2026-11-19

**Branch:** `audit/sc2-corrections-2026-11-17`  
**Scope:** Read-only arithmetic consistency of the *already frozen processed CSVs*, **not** a replay-event-to-feature reconstruction.

## Evidence and complete counts

The independent auditor streamed every row of both separate, user-supplied frozen dataset ZIP files in chunks. A 12,000-row pilot was cross-checked against a separate slower row-by-row implementation, with identical 37/43 per-feature check counts and zero mismatches on both samples.

| Data | Rows scanned | Replay IDs | Feature identities tested | Formula comparisons | Mismatches |
|---|---:|---:|---:|---:|---:|
| `v3_1_fixed` | 892,047 | 23,241 | 37 | 30,411,087 | 0 |
| `v3_2_combatfix` | 892,047 | 23,241 | 43 | 35,253,735 | 0 |
| **Combined** | **1,784,094** | — | — | **65,664,822** | **0** |

Checks used `rtol=1e-6`, `atol=1e-6` and covered intra-row differences, ratio and phase identities, worker growth, 15-second momentum and delta expressions, 60/120-second differences, and 120-second *population* standard deviations. An identity is evaluated only when the needed earlier retained CSV rows are available. This is why some temporal identities have fewer than 892,047 comparisons.

The dataset bytes match the independent B4 Git LFS freeze:
- v3.1: 268,539,879 bytes, SHA-256 `65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517`.
- v3.2: 294,197,452 bytes, SHA-256 `87ce36435e4a0fc87b8d9421291d3b1fb04d67b5187ede6a2a3b25d0dce1942e`.

Separate delivered audit artifacts:
- `sc2_b3_verify_frozen_algebra_vectorized.py` (SHA-256 `3089d2112dad42ec05a070720ff705e1ba73d8ab04733c72931adecb3cf7a73f`)
- `sc2_b3_full_algebra_2026-11-19.json` (SHA-256 `04298f64aae693eace30762f45242f0ab13af600a7a4ae40cbccf6212f0d67d3`).

These files were delivered separately for review; they are *not* embedded in the Git revision of this report. Reproduction requires both frozen full LFS payloads, Python with NumPy and pandas, and the separate auditor source.

## What this evidence supports

- All compared *derived* formulas are internally consistent for the entirety of both frozen CSVs, where their source columns and retained temporal histories permit an independent calculation.
- No discrepancies were observed in 65.7 million comparisons; no frozen files, fitted estimators, splits or historical metrics were changed.

## What it does NOT support

- It does **not** prove that replay event parsing, initial per-player combat-power quantities, counters, source event lists, or v3.2 matchup-power features were originally computed correctly.
- It does **not** establish historical executable provenance. The archived `buildf_v3_counter120.py` copy has an unterminated string literal at line 60; this is a property of the *stored copy*, not evidence about the original execution environment.
- It does **not** identify 816 replay-level exclusions. The documented 24,057 upstream replays are **not** a verified per-file raw audit; the final frozen dataset has 23,241 IDs.
- It does **not** establish numerical parity from actual `.SC2Replay` events to the final frozen feature values, nor resolve the origin of the nine v3.2-specific upstream fields.

**B3 remains OPEN.** A complete raw replay inventory, historical exclusion audit, historical original executable / bridge and raw-to-final replay-level cross-checks remain mandatory before scientific lineage sign-off.
