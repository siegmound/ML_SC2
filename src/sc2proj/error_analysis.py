from __future__ import annotations

import numpy as np
import pandas as pd


def build_prediction_frame(test_df: pd.DataFrame, y_pred: np.ndarray, y_prob: np.ndarray | None) -> pd.DataFrame:
    out = pd.DataFrame({
        'replay_id': test_df['replay_id'].astype(str).values,
        'time_sec': test_df['time_sec'].values,
        'y_true': test_df['p1_wins'].astype(int).values,
        'y_pred': np.asarray(y_pred).astype(int),
    })
    if y_prob is not None:
        out['y_prob'] = np.asarray(y_prob, dtype=float)
        out['confidence'] = np.maximum(out['y_prob'], 1.0 - out['y_prob'])
        out['uncertainty'] = np.abs(out['y_prob'] - 0.5)
    out['is_correct'] = (out['y_true'] == out['y_pred']).astype(int)
    return out


def replay_level_error_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    agg = pred_df.groupby('replay_id').agg(
        n_rows=('is_correct', 'size'),
        accuracy=('is_correct', 'mean'),
        mean_true=('y_true', 'mean'),
        mean_pred=('y_pred', 'mean'),
    ).reset_index()
    if 'y_prob' in pred_df.columns:
        prob_stats = pred_df.groupby('replay_id').agg(
            mean_prob=('y_prob', 'mean'),
            min_uncertainty=('uncertainty', 'min'),
            mean_uncertainty=('uncertainty', 'mean'),
            max_confidence=('confidence', 'max'),
        ).reset_index()
        agg = agg.merge(prob_stats, on='replay_id', how='left')
    return agg.sort_values(['accuracy', 'n_rows', 'replay_id'], ascending=[True, False, True])


def probability_flip_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    if 'y_prob' not in pred_df.columns:
        return pd.DataFrame(columns=['replay_id', 'n_flips', 'start_prob', 'end_prob'])
    rows = []
    for replay_id, part in pred_df.sort_values(['replay_id', 'time_sec']).groupby('replay_id'):
        side = (part['y_prob'] >= 0.5).astype(int).to_numpy()
        flips = int(np.sum(side[1:] != side[:-1])) if len(side) > 1 else 0
        rows.append({
            'replay_id': replay_id,
            'n_flips': flips,
            'start_prob': float(part['y_prob'].iloc[0]),
            'end_prob': float(part['y_prob'].iloc[-1]),
            'n_rows': int(len(part)),
        })
    return pd.DataFrame(rows).sort_values(['n_flips', 'n_rows'], ascending=[False, False])
