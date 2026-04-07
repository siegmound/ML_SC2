from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .deep_models import default_torch_candidates, fit_torch_candidate, predict_torch_model
from .metrics import classification_summary
from .modeling import make_numeric_preprocessor, run_group_cv_search, threshold_predictions


@dataclass
class ExperimentResult:
    model_name: str
    metrics: dict[str, Any]
    y_pred: np.ndarray
    y_prob: np.ndarray
    search_rows: list[dict[str, Any]]
    extra_artifacts: dict[str, Any]
    selection_summary: dict[str, Any]


def _subset_loaded(loaded, feature_columns: list[str]):
    if not feature_columns:
        raise ValueError('Feature selection produced an empty set.')
    clone = type(loaded)(
        feature_columns=feature_columns,
        X_train=loaded.X_train[feature_columns].copy(),
        y_train=loaded.y_train.copy(),
        groups_train=loaded.groups_train.copy(),
        X_val=loaded.X_val[feature_columns].copy(),
        y_val=loaded.y_val.copy(),
        groups_val=loaded.groups_val.copy(),
        X_test=loaded.X_test[feature_columns].copy(),
        y_test=loaded.y_test.copy(),
        groups_test=loaded.groups_test.copy(),
        train_df=loaded.train_df[['replay_id', 'time_sec', 'p1_wins', *feature_columns]].copy(),
        val_df=loaded.val_df[['replay_id', 'time_sec', 'p1_wins', *feature_columns]].copy(),
        test_df=loaded.test_df[['replay_id', 'time_sec', 'p1_wins', *feature_columns]].copy(),
    )
    for optional in ['matchup', 'race_matchup', 'map_name', 'league']:
        for frame_name in ['train_df', 'val_df', 'test_df']:
            src = getattr(loaded, frame_name)
            dst = getattr(clone, frame_name)
            if optional in src.columns and optional not in dst.columns:
                dst[optional] = src[optional].values
    return clone


def run_model_target(model_target: str, loaded, seed: int, cv_scoring: str = 'neg_log_loss', device: str = 'cpu', selection_metric: str = 'neg_log_loss', feature_columns: list[str] | None = None) -> ExperimentResult:
    subset = _subset_loaded(loaded, feature_columns or loaded.feature_columns)
    if model_target == 'logreg':
        return _run_logreg(subset, seed, cv_scoring)
    if model_target == 'rf':
        return _run_rf(subset, seed, cv_scoring)
    if model_target == 'xgb':
        return _run_xgb(subset, seed, cv_scoring, device)
    if model_target == 'mlp':
        return _run_mlp(subset, seed, cv_scoring)
    if model_target == 'mlp_torch':
        return _run_mlp_torch(subset, seed, selection_metric, device)
    raise ValueError(f'Unsupported model_target: {model_target}')


def _run_logreg(loaded, seed: int, cv_scoring: str) -> ExperimentResult:
    base = Pipeline([
        ('prep', make_numeric_preprocessor('standard')),
        ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', random_state=seed)),
    ])
    param_grid = [
        {'clf__C': 0.1, 'clf__class_weight': None},
        {'clf__C': 1.0, 'clf__class_weight': None},
        {'clf__C': 10.0, 'clf__class_weight': None},
        {'clf__C': 1.0, 'clf__class_weight': 'balanced'},
    ]
    search = run_group_cv_search(base, param_grid, loaded.X_train, loaded.y_train, loaded.groups_train, scoring=cv_scoring)
    model = search.best_estimator
    model.fit(pd.concat([loaded.X_train, loaded.X_val]), pd.concat([loaded.y_train, loaded.y_val]))
    y_prob = model.predict_proba(loaded.X_test)[:, 1]
    y_pred = threshold_predictions(y_prob)
    metrics = classification_summary(loaded.y_test, y_pred, y_prob)
    metrics.update({'best_cv_score': search.best_score, 'best_params': search.best_params})
    return ExperimentResult('logreg', metrics, y_pred, y_prob, search.cv_results, {}, {'cv_scoring': cv_scoring, 'best_params': search.best_params})


def _run_rf(loaded, seed: int, cv_scoring: str) -> ExperimentResult:
    from sklearn.inspection import permutation_importance

    base = RandomForestClassifier(random_state=seed, n_jobs=-1)
    param_grid = [
        {'n_estimators': 300, 'max_depth': 12, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None},
        {'n_estimators': 500, 'max_depth': 16, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None},
        {'n_estimators': 500, 'max_depth': None, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': None},
        {'n_estimators': 500, 'max_depth': 16, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'class_weight': 'balanced'},
    ]
    search = run_group_cv_search(base, param_grid, loaded.X_train, loaded.y_train, loaded.groups_train, scoring=cv_scoring)
    model = search.best_estimator
    model.fit(pd.concat([loaded.X_train, loaded.X_val]), pd.concat([loaded.y_train, loaded.y_val]))
    y_prob = model.predict_proba(loaded.X_test)[:, 1]
    y_pred = threshold_predictions(y_prob)
    metrics = classification_summary(loaded.y_test, y_pred, y_prob)
    metrics.update({'best_cv_score': search.best_score, 'best_params': search.best_params})
    feat = pd.DataFrame({'feature': loaded.feature_columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    perm = permutation_importance(model, loaded.X_test, loaded.y_test, n_repeats=5, random_state=seed, n_jobs=1)
    perm_df = pd.DataFrame({'feature': loaded.feature_columns, 'importance_mean': perm.importances_mean, 'importance_std': perm.importances_std}).sort_values('importance_mean', ascending=False)
    return ExperimentResult('rf', metrics, y_pred, y_prob, search.cv_results, {'feature_importance': feat, 'permutation_importance': perm_df}, {'cv_scoring': cv_scoring, 'best_params': search.best_params})


def _run_xgb(loaded, seed: int, cv_scoring: str, device: str) -> ExperimentResult:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f'xgboost not available: {exc}')

    candidates = [
        {'max_depth': 4, 'learning_rate': 0.03, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.0},
        {'max_depth': 5, 'learning_rate': 0.03, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.0},
        {'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.0},
        {'max_depth': 5, 'learning_rate': 0.05, 'min_child_weight': 3, 'subsample': 0.9, 'colsample_bytree': 0.8, 'reg_lambda': 1.5},
    ]
    rows = []
    best_score = None
    best_params = None
    for params in candidates:
        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            device=device,
            random_state=seed,
            n_estimators=2000,
            early_stopping_rounds=50,
            **params,
        )
        model.fit(loaded.X_train, loaded.y_train, eval_set=[(loaded.X_val, loaded.y_val)], verbose=False)
        y_prob_val = model.predict_proba(loaded.X_val)[:, 1]
        y_pred_val = threshold_predictions(y_prob_val)
        if cv_scoring == 'accuracy':
            score = float((loaded.y_val.to_numpy() == y_pred_val).mean())
        elif cv_scoring == 'roc_auc':
            from sklearn.metrics import roc_auc_score
            score = float(roc_auc_score(loaded.y_val, y_prob_val))
        else:
            from sklearn.metrics import log_loss
            score = float(-log_loss(loaded.y_val, y_prob_val, labels=[0, 1]))
        rows.append({**params, 'score': score, 'best_iteration': int(getattr(model, 'best_iteration', -1))})
        if best_score is None or score > best_score:
            best_score = score
            best_params = params
    assert best_params is not None and best_score is not None

    combined_X = pd.concat([loaded.X_train, loaded.X_val], ignore_index=True)
    combined_y = pd.concat([loaded.y_train, loaded.y_val], ignore_index=True)
    final_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        device=device,
        random_state=seed,
        n_estimators=2000,
        early_stopping_rounds=50,
        **best_params,
    )
    final_model.fit(loaded.X_train, loaded.y_train, eval_set=[(loaded.X_val, loaded.y_val)], verbose=False)
    y_prob = final_model.predict_proba(loaded.X_test)[:, 1]
    y_pred = threshold_predictions(y_prob)
    metrics = classification_summary(loaded.y_test, y_pred, y_prob)
    metrics.update({'best_validation_score': best_score, 'best_params': best_params, 'best_iteration': int(getattr(final_model, 'best_iteration', -1))})
    feat = pd.DataFrame({'feature': loaded.feature_columns, 'importance': final_model.feature_importances_}).sort_values('importance', ascending=False)
    return ExperimentResult('xgb', metrics, y_pred, y_prob, rows, {'feature_importance': feat}, {'cv_scoring': cv_scoring, 'best_params': best_params, 'device': device})


def _run_mlp(loaded, seed: int, cv_scoring: str) -> ExperimentResult:
    candidates = []
    for scaler in ['standard', 'quantile']:
        for hidden_layer_sizes in [(128,), (256,), (256, 128)]:
            for alpha in [1e-4, 1e-3]:
                candidates.append({
                    'prep': make_numeric_preprocessor(scaler),
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'alpha': alpha,
                    'learning_rate_init': 1e-3,
                    'batch_size': 256,
                    'max_iter': 200,
                })
    rows = []
    best_score = None
    best_bundle = None
    for candidate in candidates:
        model = Pipeline([
            ('prep', candidate['prep']),
            ('clf', MLPClassifier(hidden_layer_sizes=candidate['hidden_layer_sizes'], alpha=candidate['alpha'], learning_rate_init=candidate['learning_rate_init'], batch_size=candidate['batch_size'], max_iter=candidate['max_iter'], early_stopping=False, random_state=seed)),
        ])
        search = run_group_cv_search(model, [{}], loaded.X_train, loaded.y_train, loaded.groups_train, scoring=cv_scoring)
        score = search.best_score
        row = {k: v for k, v in candidate.items() if k != 'prep'}
        row['scaler'] = 'quantile' if candidate['prep'].named_steps['scaler'].__class__.__name__ == 'QuantileTransformer' else 'standard'
        row['score'] = score
        rows.append(row)
        if best_score is None or score > best_score:
            best_score = score
            best_bundle = candidate
    assert best_bundle is not None and best_score is not None
    model = Pipeline([
        ('prep', best_bundle['prep']),
        ('clf', MLPClassifier(hidden_layer_sizes=best_bundle['hidden_layer_sizes'], alpha=best_bundle['alpha'], learning_rate_init=best_bundle['learning_rate_init'], batch_size=best_bundle['batch_size'], max_iter=best_bundle['max_iter'], early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=seed)),
    ])
    model.fit(pd.concat([loaded.X_train, loaded.X_val]), pd.concat([loaded.y_train, loaded.y_val]))
    y_prob = model.predict_proba(loaded.X_test)[:, 1]
    y_pred = threshold_predictions(y_prob)
    metrics = classification_summary(loaded.y_test, y_pred, y_prob)
    serializable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_bundle.items() if k != 'prep'}
    serializable['scaler'] = 'quantile' if best_bundle['prep'].named_steps['scaler'].__class__.__name__ == 'QuantileTransformer' else 'standard'
    metrics.update({'best_cv_score': best_score, 'best_candidate': serializable})
    return ExperimentResult('mlp', metrics, y_pred, y_prob, rows, {}, {'cv_scoring': cv_scoring, 'best_candidate': serializable})


def _run_mlp_torch(loaded, seed: int, selection_metric: str, device: str) -> ExperimentResult:
    from sklearn.metrics import log_loss, roc_auc_score

    candidates = default_torch_candidates()
    rows = []
    best_bundle = None
    best_candidate = None
    best_score = None
    for candidate in candidates:
        bundle = fit_torch_candidate(loaded.X_train, loaded.y_train, loaded.X_val, loaded.y_val, candidate, seed=seed, device=device)
        y_prob_val = bundle['val_prob']
        y_pred_val = threshold_predictions(y_prob_val)
        if selection_metric == 'accuracy':
            score = float((loaded.y_val.to_numpy() == y_pred_val).mean())
        elif selection_metric == 'roc_auc':
            score = float(roc_auc_score(loaded.y_val, y_prob_val))
        else:
            score = float(-log_loss(loaded.y_val, y_prob_val, labels=[0, 1]))
        rows.append({**candidate.to_dict(), 'selection_metric': selection_metric, 'score': score, 'best_epoch': bundle['best_epoch'], 'best_val_loss': bundle['best_val_loss']})
        if best_score is None or score > best_score:
            best_score = score
            best_bundle = bundle
            best_candidate = candidate
    assert best_bundle is not None and best_candidate is not None and best_score is not None
    y_prob = predict_torch_model(best_bundle, loaded.X_test)
    y_pred = threshold_predictions(y_prob)
    metrics = classification_summary(loaded.y_test, y_pred, y_prob)
    metrics.update({'best_validation_score': best_score, 'best_candidate': best_candidate.to_dict(), 'best_epoch': int(best_bundle['best_epoch']), 'best_val_loss': float(best_bundle['best_val_loss']), 'device': best_bundle['device']})
    extra = {'best_training_history': pd.DataFrame(best_bundle['history'])}
    return ExperimentResult('mlp_torch', metrics, y_pred, y_prob, rows, extra, {'selection_metric': selection_metric, 'best_candidate': best_candidate.to_dict(), 'device': best_bundle['device']})
