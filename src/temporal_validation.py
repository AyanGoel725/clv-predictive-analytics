"""Rolling-origin evaluation for real dated array/account/expansion outcomes.

The supplied cross-sectional case-study CSV is deliberately rejected. Test fixtures
exercise implementation, not historical business performance.
"""
import argparse
import json
import pickle
import importlib.metadata
import platform
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import OUTPUTS, MODEL_VERSION
from src.data_cleaning import TARGET, REQUIRED, validate_raw_frame
from src.model_training import fit_pair, predict_pair, inner_predictions, fit_maps, sha256_file, SEED, DEFAULT_PARAMS
from src.backtesting import probability_metrics

TARGETS = {TARGET, 'account_cancelled_next_12m', 'capacity_expanded_next_12m'}
DATES = ['observation_date', 'feature_as_of', 'label_window_end', 'label_available_at']


def validate_history(data, target=TARGET, entity_level='array'):
    if target not in TARGETS or entity_level not in ('array', 'account'):
        raise ValueError('Unsupported dated outcome or entity level')
    if target == 'account_cancelled_next_12m' and entity_level != 'account':
        raise ValueError('Account cancellation requires account-level snapshots')
    frame = pd.read_csv(data, dtype={'account_id': str, 'array_id': str}) if isinstance(data, (str, Path)) else data.copy()
    needed = set(REQUIRED) | set(DATES) | {target}
    if entity_level == 'account':
        needed.discard('array_id')
    if target == 'capacity_expanded_next_12m':
        needed.discard('expansion_potential_signal')
    missing = needed - set(frame)
    if missing:
        raise ValueError(f'Dated validation needs explicit snapshot/outcome fields: {sorted(missing)}')
    if entity_level == 'account':
        # Internal preprocessor key only; account IDs are never model features.
        frame['array_id'] = frame['account_id']
    if frame.empty:
        raise ValueError('Historical dataset must be populated')
    if 'product_family' in frame and (frame.product_family.isna().any() or
                                      frame.product_family.astype(str).str.strip().eq('').any()):
        raise ValueError('Provided dated product families must be populated')
    frame[target] = pd.to_numeric(frame[target], errors='raise')
    if not frame[target].dropna().isin([0, 1]).all():
        raise ValueError('Outcome labels must be binary or missing')
    for date in DATES:
        frame[date] = pd.to_datetime(frame[date], errors='raise', utc=True).dt.tz_localize(None)
    if frame[['observation_date', 'feature_as_of', 'label_window_end']].isna().any().any():
        raise ValueError('Snapshot, feature-as-of and outcome-window dates are required')
    observed = frame[target].notna()
    if frame.loc[observed, 'label_available_at'].isna().any():
        raise ValueError('Observed outcomes require label availability dates')
    if (frame.feature_as_of > frame.observation_date).any():
        raise ValueError('Features must be available by the observation timestamp')
    expected_end = frame.observation_date.map(lambda value: value + pd.DateOffset(years=1))
    if not frame.label_window_end.eq(expected_end).all():
        raise ValueError('Each outcome window must end exactly 12 months after observation')
    if (frame.loc[observed, 'label_available_at'] < frame.loc[observed, 'label_window_end']).any():
        raise ValueError('Labels cannot be declared available before their window matures')
    if frame.duplicated(['account_id', 'array_id', 'observation_date']).any():
        raise ValueError('Duplicate entity/snapshot keys')
    frame[TARGET] = frame[target]
    if target == 'capacity_expanded_next_12m':
        # Prevent an externally assigned expansion tag from becoming an outcome proxy.
        frame['expansion_potential_signal'] = 'UNKNOWN'
    validated = [validate_raw_frame(group) for _, group in frame.groupby('observation_date', sort=True)]
    return pd.concat(validated, ignore_index=True)


def chronological_split(frame, cutoff, evaluation_as_of):
    cutoff, evaluation_as_of = pd.Timestamp(cutoff), pd.Timestamp(evaluation_as_of)
    if cutoff.tzinfo or evaluation_as_of.tzinfo:
        raise ValueError('Use explicit UTC-naive cutoffs matching normalized input timestamps')
    if evaluation_as_of <= cutoff:
        raise ValueError('Evaluation-as-of must follow the training cutoff')
    labeled = frame[TARGET].notna()
    train = frame.loc[labeled & (frame.observation_date < cutoff) &
                      (frame.label_window_end < cutoff) & (frame.label_available_at < cutoff)].copy()
    # Latest fully matured snapshot per entity avoids duplicate training rows and long-history overweighting.
    train = train.sort_values('observation_date').drop_duplicates(['account_id', 'array_id'], keep='last')
    validation = frame.loc[labeled & frame.observation_date.eq(cutoff) &
                           (frame.label_window_end <= evaluation_as_of) &
                           (frame.label_available_at <= evaluation_as_of)].copy()
    if train.empty or validation.empty:
        raise ValueError('Cutoff needs earlier matured training labels and matured cutoff-snapshot validation labels')
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def rolling_origin_backtest(data, cutoffs, evaluation_as_of, outcome_definition,
                            target=TARGET, entity_level='array', output_root=OUTPUTS,
                            inner_folds=3, params=None):
    if not isinstance(outcome_definition, str) or not outcome_definition.strip():
        raise ValueError('An explicit business outcome definition is required')
    frame = validate_history(data, target, entity_level)
    dates = [pd.Timestamp(value) for value in cutoffs]
    if not dates or len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError('Provide unique increasing cutoffs')
    # Validate all windows before producing a partial output directory.
    splits = [chronological_split(frame, date, evaluation_as_of) for date in dates]
    for train, _ in splits:
        if train.account_id.nunique() < inner_folds or train[TARGET].nunique() != 2:
            raise ValueError('Each training window needs enough accounts and both outcome classes')
    run_id = uuid.uuid4().hex
    root = Path(output_root) / 'temporal_runs' / run_id
    root.mkdir(parents=True, exist_ok=False)
    forecasts, metrics, windows, cohorts = [], [], [], []
    for fold, (date, (train, val)) in enumerate(zip(dates, splits), 1):
        reference = date.strftime('%Y-%m-%d')
        inner = inner_predictions(train, False, inner_folds, SEED + fold * 100,
                                  params, reference_date=reference)
        maps = fit_maps(inner, train[TARGET].to_numpy())
        pair = fit_pair(train, False, SEED + fold, params, reference_date=reference)
        raw = predict_pair(pair, val)
        scored = val[['account_id', 'array_id', 'observation_date', 'label_window_end',
                      'label_available_at', TARGET]].copy()
        scored['training_cutoff'] = date
        scored['run_id'] = run_id
        scored['prediction_source'] = 'chronological_holdout'
        if 'product_family' in val:
            scored['product_family'] = val.product_family
        scored['training_prevalence'] = float(train[TARGET].mean())
        columns = ['training_prevalence']
        if target != 'capacity_expanded_next_12m':
            scored['static_5'] = .05
            columns.append('static_5')
        for component, p in raw.items():
            name = f'{component}_sigmoid'
            scored[name] = maps[name].predict(p)
            columns.append(name)
        forecasts.append(scored)
        for model in columns:
            metrics.append({'training_cutoff': reference, 'model': model, 'rows': len(val),
                            **probability_metrics(val[TARGET].to_numpy(), scored[model].to_numpy())})
            if 'product_family' in scored:
                for family, group in scored.groupby('product_family'):
                    cohorts.append({'training_cutoff': reference, 'product_family': family,
                                    'model': model, 'rows': len(group),
                                    **probability_metrics(group[TARGET].to_numpy(), group[model].to_numpy())})
        windows.append({'cutoff': reference, 'reference_date': reference,
                        'train_rows': len(train), 'validation_rows': len(val),
                        'training_prevalence': float(train[TARGET].mean()),
                        'latest_training_label_available_at': str(train.label_available_at.max()),
                        'train_keys': train[['account_id', 'array_id', 'observation_date']].astype(str).to_dict('records'),
                        'validation_keys': val[['account_id', 'array_id', 'observation_date']].astype(str).to_dict('records')})
        with (root / f'window_{fold}_bundle.pkl').open('wb') as output:
            pickle.dump({'pair': pair, 'calibrators': maps, 'outcome_definition': outcome_definition,
                         'target': target, 'reference_date': reference}, output)
    predictions = pd.concat(forecasts, ignore_index=True)
    if entity_level == 'account':
        predictions = predictions.drop(columns='array_id')
    if target != TARGET:
        predictions = predictions.rename(columns={TARGET: target})
    predictions.to_csv(root / 'historical_predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(root / 'historical_metrics.csv', index=False)
    if cohorts:
        pd.DataFrame(cohorts).to_csv(root / 'historical_product_cohorts.csv', index=False)
    frame.to_csv(root / 'validated_history.csv', index=False)
    manifest = {
        'run_id': run_id, 'model_version': MODEL_VERSION, 'target': target, 'entity_level': entity_level,
        'python_version': platform.python_version(), 'seed': SEED, 'inner_folds': inner_folds,
        'parameters': {**DEFAULT_PARAMS, **(params or {})},
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost']},
        'outcome_definition': outcome_definition, 'evaluation_as_of': str(pd.Timestamp(evaluation_as_of)),
        'validation_design': 'rolling_origin_matured_labels_latest_training_snapshot_per_entity',
        'inner_calibration_design': 'account_grouped_training_oof', 'windows': windows,
        'account_overlap_policy': 'Existing accounts may recur in later windows; IDs are excluded from features',
        'feature_availability_policy': 'Input feature_as_of must not exceed observation_date; caller must supply genuine as-of values',
        'artifacts': {p.name: sha256_file(p) for p in root.iterdir() if p.is_file()},
        'source_sha256': {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return root, predictions, pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--cutoffs', required=True, help='Comma-separated observation timestamps')
    parser.add_argument('--evaluation-as-of', required=True)
    parser.add_argument('--outcome-definition', required=True)
    parser.add_argument('--target', choices=sorted(TARGETS), default=TARGET)
    parser.add_argument('--entity-level', choices=['array', 'account'], default='array')
    args = parser.parse_args()
    root, _, _ = rolling_origin_backtest(args.data, args.cutoffs.split(','), args.evaluation_as_of,
                                         args.outcome_definition, args.target, args.entity_level)
    print(f'Chronological evidence saved: {root}')


if __name__ == '__main__':
    main()
