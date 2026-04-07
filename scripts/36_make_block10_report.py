from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    detail_path = PROJECT_ROOT / 'results' / 'block10_candidate' / 'block10_logreg_multiseed.csv'
    leaderboard_path = PROJECT_ROOT / 'results' / 'block10_candidate' / 'block10_candidate_leaderboard.csv'
    status_path = PROJECT_ROOT / 'results' / 'block10_status_report.json'
    if not detail_path.exists() or not leaderboard_path.exists():
        raise SystemExit('Block 10 benchmark outputs not found.')
    detail = pd.read_csv(detail_path)
    leaderboard = pd.read_csv(leaderboard_path)
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    report = {
        'dataset_name': detail['dataset_name'].iloc[0],
        'n_rows': int(detail['n_rows'].iloc[0]),
        'n_replays': int(detail['n_replays'].iloc[0]),
        'completed_models': status.get('completed_models', ['logreg']),
        'attempted_models': status.get('attempted_models', ['logreg']),
        'summary': leaderboard.to_dict(orient='records'),
        'seeds': detail['seed'].tolist(),
    }
    out = PROJECT_ROOT / 'results' / 'block10_candidate' / 'block10_report.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
