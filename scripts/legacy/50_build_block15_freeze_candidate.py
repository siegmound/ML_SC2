import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUT = RESULTS / 'block15_freeze'
OUT.mkdir(parents=True, exist_ok=True)

block14 = json.loads((RESULTS / 'block14_report' / 'block14_report.json').read_text())
block12 = json.loads((RESULTS / 'block12_report' / 'block12_report.json').read_text())
block13 = json.loads((RESULTS / 'block13_report' / 'block13_report.json').read_text())

winner = block14['datasets']['real_v3_2_combatfix_smoke3000']['best_profile']
selection = {
    'selected_dataset': 'real_v3_2_combatfix_smoke3000',
    'selected_model_family': 'random_forest',
    'selected_profile': 'no_counter',
    'selection_basis': {
        'block11': 'RF beat XGB on real_v3_1_fixed_smoke3000 across seeds 42-44',
        'block12': 'economy/scouting/composition were the most useful families; counter features looked suspicious',
        'block13': 'combatfix and no_counter improved seed-42 compact RF profiles',
        'block14': 'combatfix + no_counter won the multi-seed compact RF comparison',
    },
    'metrics_on_selected_regime': winner,
    'context': {
        'block12_report_exists': bool(block12),
        'block13_report_exists': bool(block13),
    },
}
(OUT / 'selected_freeze_candidate.json').write_text(json.dumps(selection, indent=2))

rows = []
for dataset_name, payload in block14['datasets'].items():
    for row in payload['rows']:
        r = {'dataset_name': dataset_name}
        r.update(row)
        rows.append(r)
summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT / 'block15_final_profile_summary.csv', index=False)
leaderboard = summary_df.sort_values(['dataset_name', 'accuracy_mean', 'roc_auc_mean'], ascending=[True, False, False])
leaderboard.to_csv(OUT / 'block15_leaderboard.csv', index=False)

md_lines = [
    '# Block 15 Freeze Candidate',
    '',
    'Selected candidate: **RF + real_v3_2_combatfix_smoke3000 + no_counter**',
    '',
    '## Why this candidate',
]
for k, v in selection['selection_basis'].items():
    md_lines.append(f'- **{k}**: {v}')
md_lines.extend(['', '## Selected metrics'])
for k, v in winner.items():
    md_lines.append(f'- **{k}**: {v}')
(OUT / 'block15_freeze_candidate.md').write_text('\n'.join(md_lines))
print('Wrote block15 freeze candidate artifacts to', OUT)
