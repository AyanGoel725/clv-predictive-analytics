"""Account-grouped evaluation, inner-OOF calibration, and reusable inference bundles."""
import hashlib
import importlib.metadata
import json
import pickle
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import OUTPUTS, MODEL_VERSION, PRIMARY_SCORE, REFERENCE_DATE
from src.data_cleaning import TARGET, validate_raw_frame
from src.feature_engineering import CLVPreprocessor

SEED = 42
DEFAULT_PARAMS = {'xgb_estimators': 150, 'rf_estimators': 200}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_hash(df):
    """Hash raw labeled inputs in canonical column order, including row order."""
    cols = ['account_id', 'array_id', 'contract_end_date',
            'historical_15yr_net_profit_usd', 'current_capacity_tb',
            'utilization_rate_pct', 'expansion_potential_signal', TARGET]
    return hashlib.sha256(df[cols].to_csv(index=False, date_format='%Y-%m-%d').encode()).hexdigest()


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class ProbabilityMap:
    """Sigmoid or isotonic mapping fitted exclusively to inner OOF probabilities."""
    def __init__(self, method='sigmoid'):
        self.method = method

    def fit(self, p, y):
        if np.unique(y).size != 2:
            raise ValueError('Calibration requires both target classes')
        if self.method == 'sigmoid':
            self.estimator_ = LogisticRegression(C=1e6, max_iter=1000, random_state=SEED)
            self.estimator_.fit(logit(p), y)
        elif self.method == 'isotonic':
            self.estimator_ = IsotonicRegression(out_of_bounds='clip').fit(p, y)
        else:
            raise ValueError('Mapping must be sigmoid or isotonic')
        return self

    def predict(self, p):
        if self.method == 'sigmoid':
            return self.estimator_.predict_proba(logit(p))[:, 1]
        return self.estimator_.predict(p)


def grouped_splits(df, n_splits):
    groups = df.account_id.to_numpy()
    if df.account_id.nunique() < n_splits:
        raise ValueError('Insufficient distinct accounts for grouped folds')
    for train, val in GroupKFold(n_splits=n_splits).split(df, df[TARGET], groups):
        if set(groups[train]) & set(groups[val]):
            raise AssertionError('Account overlap between training and validation')
        if df.iloc[train][TARGET].nunique() != 2:
            raise ValueError('Each training fold requires both target classes')
        yield train, val


def fit_pair(train, weighted, seed, params=None, reference_date=REFERENCE_DATE):
    params = {**DEFAULT_PARAMS, **(params or {})}
    prep = CLVPreprocessor(reference_date=reference_date).fit(train)
    X = prep.transform(train)
    y = train[TARGET].to_numpy(dtype=int)
    positive_weight = (y == 0).sum() / (y == 1).sum() if weighted else 1.0
    xgb = XGBClassifier(
        n_estimators=params['xgb_estimators'], max_depth=4, learning_rate=.05,
        subsample=.8, colsample_bytree=.8, reg_alpha=.1, reg_lambda=1.,
        scale_pos_weight=positive_weight, eval_metric='logloss',
        random_state=seed, n_jobs=2,
    ).fit(X, y)
    rf = RandomForestClassifier(
        n_estimators=params['rf_estimators'], max_depth=8,
        min_samples_split=10, min_samples_leaf=4, max_features='sqrt',
        class_weight='balanced' if weighted else None,
        random_state=seed, n_jobs=2,
    ).fit(X, y)
    return {'preprocessor': prep, 'xgb': xgb, 'rf': rf}


def predict_pair(pair, data):
    X = pair['preprocessor'].transform(data)
    a = pair['xgb'].predict_proba(X)[:, 1]
    b = pair['rf'].predict_proba(X)[:, 1]
    return {'xgb': a, 'rf': b, 'ensemble': .6 * a + .4 * b}


def inner_predictions(train, weighted, n_splits, seed, params=None, reference_date=REFERENCE_DATE):
    inner = {name: np.full(len(train), np.nan) for name in ('xgb', 'rf', 'ensemble')}
    for fold, (tr, va) in enumerate(grouped_splits(train, n_splits), 1):
        pair = fit_pair(train.iloc[tr], weighted, seed + fold, params, reference_date=reference_date)
        pred = predict_pair(pair, train.iloc[va])
        for name in inner:
            inner[name][va] = pred[name]
    if not all(np.isfinite(p).all() for p in inner.values()):
        raise AssertionError('Incomplete inner out-of-fold predictions')
    return inner


def fit_maps(predictions, y):
    return {f'{name}_{method}': ProbabilityMap(method).fit(p, y)
            for name, p in predictions.items() for method in ('sigmoid', 'isotonic')}


def predict_bundle(bundle, raw):
    pred = predict_pair(bundle, raw)
    return {
        'xgb': bundle['calibrators']['xgb_sigmoid'].predict(pred['xgb']),
        'rf': bundle['calibrators']['rf_sigmoid'].predict(pred['rf']),
        'ensemble': bundle['calibrators']['ensemble_sigmoid'].predict(pred['ensemble']),
        'prevalence': np.full(len(raw), bundle['training_prevalence']),
    }


def train_and_evaluate_models(raw, n_splits=5, inner_splits=3, output_root=OUTPUTS, params=None):
    """Evaluate prespecified candidates; no winner or threshold selected on outer labels."""
    raw = validate_raw_frame(raw, require_target=True)
    labeled = raw.loc[raw[TARGET].notna()].reset_index(drop=True)
    if labeled.empty:
        raise ValueError('Training needs labeled rows')
    labeled[TARGET] = labeled[TARGET].astype(int)
    run_id = uuid.uuid4().hex
    run_dir = Path(output_root) / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    y = labeled[TARGET].to_numpy()
    columns = ['account_id', 'array_id', 'contract_end_date',
               'historical_15yr_net_profit_usd', 'expansion_potential_signal', TARGET]
    evidence = labeled[columns].copy()
    evidence['prediction_source'] = 'grouped_oof'
    evidence['run_id'] = run_id
    evidence['fold_id'] = -1
    evidence['static_5'] = .05
    evidence['training_prevalence'] = np.nan
    evidence['logistic'] = np.nan
    for weight in ('weighted', 'unweighted'):
        for name in ('xgb', 'rf', 'ensemble'):
            for mapping in ('raw', 'sigmoid', 'isotonic'):
                evidence[f'{weight}_{name}_{mapping}'] = np.nan
    folds = []
    for fold, (tr, va) in enumerate(grouped_splits(labeled, n_splits), 1):
        train, val = labeled.iloc[tr], labeled.iloc[va]
        print(f'Outer account fold {fold}/{n_splits}: train={len(tr)}, validation={len(va)}', flush=True)
        evidence.loc[va, 'fold_id'] = fold
        evidence.loc[va, 'training_prevalence'] = float(train[TARGET].mean())
        prep = CLVPreprocessor().fit(train)
        logistic = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=SEED))
        logistic.fit(prep.transform(train), train[TARGET])
        evidence.loc[va, 'logistic'] = logistic.predict_proba(prep.transform(val))[:, 1]
        folds.append({
            'fold_id': fold, 'train_accounts': sorted(train.account_id.unique().tolist()),
            'validation_accounts': sorted(val.account_id.unique().tolist()),
            'train_rows': len(tr), 'validation_rows': len(va),
            'capacity_edges': [None if not np.isfinite(x) else float(x) for x in prep.capacity_edges_],
            'training_prevalence': float(train[TARGET].mean()),
        })
        for weighted in (True, False):
            prefix = 'weighted' if weighted else 'unweighted'
            inner = inner_predictions(train.reset_index(drop=True), weighted, inner_splits, SEED + 100 * fold, params)
            maps = fit_maps(inner, train[TARGET].to_numpy())
            pair = fit_pair(train, weighted, SEED + fold, params)
            pred = predict_pair(pair, val)
            for name, p in pred.items():
                evidence.loc[va, f'{prefix}_{name}_raw'] = p
                for method in ('sigmoid', 'isotonic'):
                    evidence.loc[va, f'{prefix}_{name}_{method}'] = maps[f'{name}_{method}'].predict(p)
    # Final inference mapping uses grouped inner OOF on all labeled data.
    print('Fitting final unweighted inference bundle and grouped calibrators', flush=True)
    final_inner = inner_predictions(labeled, False, inner_splits, SEED + 1000, params)
    bundle = fit_pair(labeled, False, SEED, params)
    bundle.update(
        calibrators=fit_maps(final_inner, y), training_prevalence=float(y.mean()),
        model_version=MODEL_VERSION, training_run_id=run_id,
        primary_score=PRIMARY_SCORE, target=TARGET,
        reference_date=bundle['preprocessor'].reference_date,
        shap_units='xgb_raw_margin_log_odds',
    )
    bundle_path = run_dir / 'inference_bundle.pkl'
    with bundle_path.open('wb') as f:
        pickle.dump(bundle, f)
    evidence_path = run_dir / 'evaluation_predictions.csv'
    evidence.to_csv(evidence_path, index=False, date_format='%Y-%m-%d')
    snapshot_path = run_dir / 'training_snapshot.csv'
    raw.to_csv(snapshot_path, index=False, date_format='%Y-%m-%d')
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'shap', 'matplotlib', 'tabulate')}
    manifest = {
        'run_id': run_id, 'model_version': MODEL_VERSION,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(), 'platform': platform.platform(),
        'validation_design': 'account_grouped_outer_cv_inner_oof_calibration',
        'prediction_source': 'grouped_oof', 'outer_folds': n_splits,
        'inner_folds': inner_splits, 'seed': SEED, 'parameters': {**DEFAULT_PARAMS, **(params or {})},
        'labeled_rows': len(labeled), 'unlabeled_rows': len(raw) - len(labeled),
        'labeled_input_sha256': frame_hash(labeled),
        'evaluation_sha256': sha256_file(evidence_path), 'bundle_sha256': sha256_file(bundle_path),
        'training_snapshot_sha256': sha256_file(snapshot_path),
        'primary_score': PRIMARY_SCORE,
        'candidate_policy': 'Prespecified unweighted sigmoid ensemble; not a selected champion',
        'feature_names': bundle['preprocessor'].feature_names_,
        'versions': versions, 'folds': folds,
        'source_sha256': {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
        'warehouse_source_sha256': {p.name: sha256_file(p) for p in (Path(__file__).parent.parent / 'sql').glob('*.sql')},
        'limitations': ['No observation timestamps or verified label maturity',
                        'No untouched final test or demonstrated account cancellation labels',
                        'No product family, expansion outcomes, or future cash-flow labels',
                        'Live Snowflake execution unverified'],
    }
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # Only update latest pointer after the complete artifact set has been saved.
    (Path(output_root) / 'latest_run.json').write_text(
        json.dumps({'run_id': run_id}, indent=2), encoding='utf-8')
    return {'run_dir': run_dir, 'manifest': manifest, 'evaluation': evidence, 'bundle': bundle}


def latest_run(output_root=OUTPUTS):
    root = Path(output_root)
    pointer = json.loads((root / 'latest_run.json').read_text(encoding='utf-8'))
    return root / 'runs' / pointer['run_id']


def load_bundle(run_dir=None):
    run_dir = Path(run_dir) if run_dir is not None else latest_run()
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    path = run_dir / 'inference_bundle.pkl'
    if sha256_file(path) != manifest['bundle_sha256']:
        raise ValueError('Inference bundle hash does not match manifest')
    with path.open('rb') as f:
        bundle = pickle.load(f)
    if bundle['training_run_id'] != manifest['run_id'] or bundle['model_version'] != manifest['model_version']:
        raise ValueError('Bundle lineage mismatch')
    return bundle, manifest


if __name__ == '__main__':
    from src.pipeline import main
    main(['train'])

