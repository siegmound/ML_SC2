
# Block 15

Block 15 turns the previous benchmark work into a freeze-candidate package.

## Goal

Freeze one primary candidate for the remake and export compact artifacts that are usable for:
- paper tables
- presentation updates
- repository freeze notes
- final audit

## Selected candidate

**Random Forest + `real_v3_2_combatfix_smoke3000` + `no_counter`**

## Why

- Block 11: RF beat XGB on `real_v3_1_fixed_smoke3000` across seeds 42–44.
- Block 12: economy/scouting/composition looked useful; counter features looked suspicious.
- Block 13: on seed 42, `v3_2_combatfix` and `no_counter` improved compact RF profile results.
- Block 14: across seeds 42–44, `v3_2_combatfix + no_counter` was the strongest compact RF profile.

## Block 15 deliverables

- final candidate JSON
- final summary CSV
- paper-ready CSV/TEX tables
- final report JSON
- concise markdown summary
