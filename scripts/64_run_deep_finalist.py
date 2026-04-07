
import argparse
import json
import math
import random
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

TARGET = 'p1_wins'
GROUP = 'replay_id'
TIME = 'time_sec'
DROP_ALWAYS = {TARGET, GROUP, TIME}
COUNTER_TOKENS = ('counter',)
LOSSES_TOKENS = ('loss', 'killed', 'destroyed', 'recent_loss')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-zip', required=True)
    ap.add_argument('--split-json', required=True)
    ap.add_argument('--dataset-name', required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--candidate-set', choices=['strict'], default='strict')
    ap.add_argument('--selection-metric', choices=['neg_log_loss', 'roc_auc'], default='neg_log_loss')
    ap.add_argument('--output-root', default='results/deep_final')
    ap.add_argument('--random-state', type=int, default=42)
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_zip_csv(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = [n for n in zf.namelist() if n.lower().endswith('.csv')]
        if not names:
            raise FileNotFoundError(f'No CSV found in {zip_path}')
        with zf.open(names[0]) as f:
            return pd.read_csv(f)


def select_features(columns, profile):
    feats = [c for c in columns if c not in DROP_ALWAYS]
    if profile == 'full':
        return feats
    out = []
    for c in feats:
        cl = c.lower()
        if profile in ('no_counter', 'no_counter_no_losses') and any(tok in cl for tok in COUNTER_TOKENS):
            continue
        if profile == 'no_counter_no_losses' and any(tok in cl for tok in LOSSES_TOKENS):
            continue
        out.append(c)
    return out


def get_candidates():
    return [
        {
            'name': 'std_relu_256_128_full',
            'profile': 'full',
            'scaler': 'standard',
            'hidden_dims': [256, 128],
            'dropout': 0.25,
            'activation': 'relu',
            'batch_norm': True,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 512,
            'max_epochs': 80,
            'patience': 10,
            'scheduler': True,
            'use_pos_weight': True,
        },
        {
            'name': 'std_lrelu_256_128_full',
            'profile': 'full',
            'scaler': 'standard',
            'hidden_dims': [256, 128],
            'dropout': 0.25,
            'activation': 'lrelu',
            'batch_norm': True,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 512,
            'max_epochs': 80,
            'patience': 10,
            'scheduler': True,
            'use_pos_weight': True,
        },
        {
            'name': 'qt_lrelu_256_128_full',
            'profile': 'full',
            'scaler': 'quantile',
            'hidden_dims': [256, 128],
            'dropout': 0.25,
            'activation': 'lrelu',
            'batch_norm': True,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 512,
            'max_epochs': 80,
            'patience': 10,
            'scheduler': True,
            'use_pos_weight': True,
        },
        {
            'name': 'qt_lrelu_256_128_nocounter',
            'profile': 'no_counter',
            'scaler': 'quantile',
            'hidden_dims': [256, 128],
            'dropout': 0.25,
            'activation': 'lrelu',
            'batch_norm': True,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 512,
            'max_epochs': 80,
            'patience': 10,
            'scheduler': True,
            'use_pos_weight': True,
        },
        {
            'name': 'std_relu_512_256_nocounter',
            'profile': 'no_counter',
            'scaler': 'standard',
            'hidden_dims': [512, 256],
            'dropout': 0.30,
            'activation': 'relu',
            'batch_norm': True,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 512,
            'max_epochs': 80,
            'patience': 10,
            'scheduler': True,
            'use_pos_weight': True,
        },
        {
            'name': 'qt_relu_256_128_nocounter',
            'profile': 'no_counter',
            'scaler': 'quantile',
            'hidden_dims': [256, 128],
            'dropout': 0.20,
            'activation': 'relu',
            'batch_norm': True,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 512,
            'max_epochs': 80,
            'patience': 10,
            'scheduler': True,
            'use_pos_weight': True,
        },
    ]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, dropout: float, activation: str, batch_norm: bool):
        super().__init__()
        layers = []
        prev = input_dim
        act_layer = nn.ReLU if activation == 'relu' else lambda: nn.LeakyReLU(negative_slope=0.01)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_layer())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def make_scaler(kind: str):
    if kind == 'standard':
        return StandardScaler()
    if kind == 'quantile':
        return QuantileTransformer(output_distribution='normal', random_state=42)
    raise ValueError(kind)


def to_writable_float32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.astype(np.float32, copy=False)).copy()


def to_writable_int64(y: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(y.astype(np.int64, copy=False)).copy()


def build_loaders(xtr, ytr, xva, yva, batch_size: int):
    xtr = to_writable_float32(xtr)
    ytr = to_writable_int64(ytr)
    xva = to_writable_float32(xva)
    yva = to_writable_int64(yva)

    train_ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    val_ds = TensorDataset(torch.from_numpy(xva), torch.from_numpy(yva))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def predict_prob(model, x: np.ndarray, device: torch.device, batch_size: int = 4096):
    model.eval()
    x = to_writable_float32(x)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            outs.append(prob)
    return np.concatenate(outs, axis=0)


def metrics_from_probs(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'log_loss': float(log_loss(y_true, y_prob)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def make_prediction_frame(df, y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return pd.DataFrame({
        GROUP: df[GROUP].values,
        TIME: df[TIME].values,
        'y_true': y_true.values if hasattr(y_true, 'values') else y_true,
        'y_prob': y_prob,
        'y_pred': y_pred,
    })


def run_candidate(train_df, val_df, test_df, candidate, device, selection_metric, seed):
    features = select_features(train_df.columns, candidate['profile'])

    xtr_raw = train_df[features].to_numpy()
    ytr = train_df[TARGET].astype(np.int64).to_numpy()
    xva_raw = val_df[features].to_numpy()
    yva = val_df[TARGET].astype(np.int64).to_numpy()
    xte_raw = test_df[features].to_numpy()
    yte = test_df[TARGET].astype(np.int64).to_numpy()

    scaler = make_scaler(candidate['scaler'])
    xtr = scaler.fit_transform(xtr_raw)
    xva = scaler.transform(xva_raw)
    xte = scaler.transform(xte_raw)

    train_loader, val_loader = build_loaders(xtr, ytr, xva, yva, candidate['batch_size'])
    model = MLP(len(features), candidate['hidden_dims'], candidate['dropout'], candidate['activation'], candidate['batch_norm']).to(device)

    pos_weight = None
    if candidate['use_pos_weight']:
        pos = max(1, int(ytr.sum()))
        neg = max(1, int((1 - ytr).sum()))
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=candidate['lr'], weight_decay=candidate['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) if candidate['scheduler'] else None

    best_state = None
    best_val_loss = math.inf
    epochs_no_improve = 0
    history = []

    for epoch in range(1, candidate['max_epochs'] + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.float().to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.detach().cpu().item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        if scheduler is not None:
            scheduler.step(val_loss)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': float(optimizer.param_groups[0]['lr']),
        })

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= candidate['patience']:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_prob = predict_prob(model, xva, device)
    test_prob = predict_prob(model, xte, device)
    val_metrics = metrics_from_probs(yva, val_prob)
    test_metrics = metrics_from_probs(yte, test_prob)

    if selection_metric == 'neg_log_loss':
        selection_score = -val_metrics['log_loss']
    elif selection_metric == 'roc_auc':
        selection_score = val_metrics['roc_auc']
    else:
        raise ValueError(selection_metric)

    return {
        'candidate': candidate,
        'features': features,
        'scaler': scaler,
        'history': history,
        'best_model_state': deepcopy(model.state_dict()),
        'selection_score': float(selection_score),
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'validation_predictions': (yva, val_prob),
        'test_predictions': (yte, test_prob),
        'test_df': test_df[[GROUP, TIME]].copy(),
        'val_df': val_df[[GROUP, TIME]].copy(),
    }


def run():
    args = parse_args()
    set_seed(args.random_state)
    dataset_zip = Path(args.dataset_zip)
    split_json = Path(args.split_json)

    with open(split_json, 'r', encoding='utf-8') as f:
        split = json.load(f)
    seed = split.get('seed', args.random_state)

    out_dir = Path(args.output_root) / args.dataset_name / f'seed_{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    print(f'Loading dataset from {dataset_zip} ...')
    df = load_zip_csv(dataset_zip)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    print(f'Loaded rows={len(df)} unique_replays={df[GROUP].nunique()}')

    train_groups = set(split['train_groups'])
    val_groups = set(split['val_groups'])
    test_groups = set(split['test_groups'])

    train_df = df[df[GROUP].isin(train_groups)].copy()
    val_df = df[df[GROUP].isin(val_groups)].copy()
    test_df = df[df[GROUP].isin(test_groups)].copy()
    print(f'Split rows train={len(train_df)} val={len(val_df)} test={len(test_df)} device={device.type}')

    candidates = get_candidates()
    rows = []
    best = None

    for i, candidate in enumerate(candidates, start=1):
        print(f'Candidate {i}/{len(candidates)}: {candidate}')
        result = run_candidate(train_df, val_df, test_df, candidate, device, args.selection_metric, seed)
        rows.append({
            'seed': seed,
            'candidate_name': candidate['name'],
            'profile': candidate['profile'],
            'selection_score': result['selection_score'],
            'val_accuracy': result['validation_metrics']['accuracy'],
            'val_balanced_accuracy': result['validation_metrics']['balanced_accuracy'],
            'val_roc_auc': result['validation_metrics']['roc_auc'],
            'val_log_loss': result['validation_metrics']['log_loss'],
            'test_accuracy': result['test_metrics']['accuracy'],
            'test_balanced_accuracy': result['test_metrics']['balanced_accuracy'],
            'test_roc_auc': result['test_metrics']['roc_auc'],
            'test_log_loss': result['test_metrics']['log_loss'],
            'n_features': len(result['features']),
        })
        if best is None or result['selection_score'] > best['selection_score']:
            best = result

    search_df = pd.DataFrame(rows).sort_values('selection_score', ascending=False).reset_index(drop=True)
    search_df.to_csv(out_dir / 'validation_search_results.csv', index=False)

    best_history = pd.DataFrame(best['history'])
    best_history.to_csv(out_dir / 'best_training_history.csv', index=False)

    val_y, val_prob = best['validation_predictions']
    test_y, test_prob = best['test_predictions']

    val_preds = make_prediction_frame(
        pd.concat([best['val_df'], pd.DataFrame({'y_true': val_y, 'y_prob': val_prob})], axis=1),
        pd.Series(val_y),
        val_prob,
    )
    test_preds = make_prediction_frame(
        pd.concat([best['test_df'], pd.DataFrame({'y_true': test_y, 'y_prob': test_prob})], axis=1),
        pd.Series(test_y),
        test_prob,
    )

    # cleaner prediction frames
    val_preds = pd.DataFrame({
        GROUP: best['val_df'][GROUP].values,
        TIME: best['val_df'][TIME].values,
        'y_true': val_y,
        'y_prob': val_prob,
        'y_pred': (val_prob >= 0.5).astype(np.int32),
    })
    test_preds = pd.DataFrame({
        GROUP: best['test_df'][GROUP].values,
        TIME: best['test_df'][TIME].values,
        'y_true': test_y,
        'y_prob': test_prob,
        'y_pred': (test_prob >= 0.5).astype(np.int32),
    })

    summary = {
        'dataset_name': args.dataset_name,
        'seed': seed,
        'model': 'deep_final',
        'selection_metric': args.selection_metric,
        'device': device.type,
        'best_candidate': best['candidate']['name'],
        'best_profile': best['candidate']['profile'],
        'n_rows_train': int(len(train_df)),
        'n_rows_val': int(len(val_df)),
        'n_rows_test': int(len(test_df)),
        'n_features': int(len(best['features'])),
        'validation_metrics': best['validation_metrics'],
        'test_metrics': best['test_metrics'],
    }

    with open(out_dir / 'metrics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    val_preds.to_csv(out_dir / 'validation_predictions.csv', index=False)
    test_preds.to_csv(out_dir / 'test_predictions.csv', index=False)

    with open(out_dir / 'config_used.json', 'w', encoding='utf-8') as f:
        json.dump({
            'best_candidate': best['candidate'],
            'features': best['features'],
            'selection_metric': args.selection_metric,
        }, f, indent=2)

    with open(out_dir / 'artifacts_manifest.json', 'w', encoding='utf-8') as f:
        json.dump({
            'metrics_summary': str(out_dir / 'metrics_summary.json'),
            'validation_search_results': str(out_dir / 'validation_search_results.csv'),
            'best_training_history': str(out_dir / 'best_training_history.csv'),
            'validation_predictions': str(out_dir / 'validation_predictions.csv'),
            'test_predictions': str(out_dir / 'test_predictions.csv'),
            'config_used': str(out_dir / 'config_used.json'),
        }, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f'Saved outputs to {out_dir}')


if __name__ == '__main__':
    run()
