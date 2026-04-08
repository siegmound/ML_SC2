
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'block15_freeze'
OUT.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(OUT / 'block15_final_profile_summary.csv')
selected = summary[summary['dataset_name'].isin(['real_v3_1_fixed_smoke3000','real_v3_2_combatfix_smoke3000'])].copy()
cols = ['dataset_name','profile','accuracy_mean','accuracy_std','roc_auc_mean','roc_auc_std','log_loss_mean','log_loss_std','n_runs','n_features_mean']
selected = selected[cols]
selected.to_csv(OUT / 'block15_paper_table_profiles.csv', index=False)
selected.to_latex(OUT / 'block15_paper_table_profiles.tex', index=False, float_format='%.4f')

# winner-only table
winner = selected[(selected['dataset_name']=='real_v3_2_combatfix_smoke3000') & (selected['profile']=='no_counter')]
winner.to_csv(OUT / 'block15_paper_table_winner.csv', index=False)
winner.to_latex(OUT / 'block15_paper_table_winner.tex', index=False, float_format='%.4f')
print('Exported block15 paper tables')
