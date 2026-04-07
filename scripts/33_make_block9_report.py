from pathlib import Path
import json, pandas as pd
root=Path('/mnt/data/ml_sc2_remake/results/block9_multiseed')
frames=[]
for p in sorted(root.glob('*_multiseed_detailed.csv')):
    frames.append(pd.read_csv(p))
if frames:
    df=pd.concat(frames, ignore_index=True)
    df.to_csv(root/'block9_all_detailed.csv', index=False)
    agg=df.groupby('model_name').agg(n_runs=('seed','count'), accuracy_mean=('accuracy','mean'), accuracy_std=('accuracy','std'), balanced_accuracy_mean=('balanced_accuracy','mean'), balanced_accuracy_std=('balanced_accuracy','std'), roc_auc_mean=('roc_auc','mean'), roc_auc_std=('roc_auc','std'), log_loss_mean=('log_loss','mean'), log_loss_std=('log_loss','std')).reset_index().sort_values('accuracy_mean', ascending=False)
    agg.to_csv(root/'block9_model_leaderboard.csv', index=False)
    report={'files':[p.name for p in sorted(root.iterdir())], 'leaderboard': agg.to_dict(orient='records')}
else:
    report={'files':[], 'leaderboard':[]}
Path('/mnt/data/ml_sc2_remake/results/block9_status_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
