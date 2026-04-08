# Block 13

Goal: stress-test the current Random Forest winner with targeted, interpretable profiles instead of broader re-tuning.

Main profile set:
- full
- no_counter
- no_losses
- no_counter_no_losses
- economy_only
- economy_scouting
- economy_scouting_combat
- economy_scouting_composition

This block is meant to answer:
1. Does RF improve when dropping the suspicious families from block 12?
2. How much of RF performance survives with a simpler feature set?
3. Does the same profile ranking persist across dataset variants such as v3_1_fixed and v3_2_combatfix?
