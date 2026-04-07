
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUT = RESULTS / 'block15_freeze'
OUT.mkdir(parents=True, exist_ok=True)

selected = json.loads((OUT / 'selected_freeze_candidate.json').read_text())
block14 = json.loads((RESULTS / 'block14_report' / 'block14_report.json').read_text())
block11 = pd.read_csv(RESULTS / 'block11_imported' / 'imported_metrics.csv')

report = {
    'selected_candidate': selected,
    'rf_vs_xgb_block11_means': block11.groupby('model')[['accuracy','balanced_accuracy','roc_auc','log_loss']].mean().reset_index().to_dict(orient='records'),
    'block14_cross_dataset_winner': block14['cross_dataset_winner'],
    'final_recommendation': 'Freeze the remake around RF + real_v3_2_combatfix_smoke3000 + no_counter, then only reopen if a larger staged run or full-data run overturns it.'
}
(OUT / 'block15_report.json').write_text(json.dumps(report, indent=2))
print('Wrote', OUT / 'block15_report.json')
