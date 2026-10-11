"""Account array-event summaries, capacity diagnostics and explicit horizon scenarios."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import PRIMARY_SCORE
from src.data_cleaning import TARGET, validate_raw_frame
from src.model_training import load_bundle, predict_bundle, sha256_file, latest_run
from src.backtesting import read_evidence, probability_metrics, paired_account_ci
from src.scenarios import survival_scenario

KEYS = ['account_id', 'array_id']


def input_fingerprint(value):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return sha256_file(value)
    return hashlib.sha256(value.to_csv(index=False).encode('utf-8')).hexdigest()


def account_summary(raw, predictions, probability_column, source):
    """Bounds concern array events conditional on estimated marginals, not cancellation.

    Complete array coverage is required for bounds; missing labels never become zero.
    No independence assumption or account-cancellation target is silently introduced.
    """
    raw = validate_raw_frame(raw)
    p = predictions[KEYS + [probability_column]].copy()
    if p.duplicated(KEYS).any() or p[KEYS].isna().any().any():
        raise ValueError('Prediction keys must be populated and unique')
    values = p[probability_column].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Account aggregation requires finite array probabilities in [0,1]')
    matched = p.merge(raw[KEYS], on=KEYS, how='left', indicator=True, validate='one_to_one')
    if matched['_merge'].ne('both').any():
        raise ValueError('Prediction contains an unknown array key')
    merged = raw.merge(p.rename(columns={probability_column: '_p'}), on=KEYS,
                       how='left', validate='one_to_one')
    records = []
    for account, group in merged.groupby('account_id', sort=True):
        probs = group['_p'].dropna().to_numpy()
        n, covered = len(group), len(probs)
        complete = n == covered
        known_labels = group[TARGET].notna().all()
        records.append({
            'account_id': account, 'arrays_expected': n, 'arrays_scored': covered,
            'complete_probability_coverage': complete,
            'mean_scored_array_event_probability': float(probs.mean()) if covered else np.nan,
            'max_scored_array_event_probability': float(probs.max()) if covered else np.nan,
            'any_array_event_probability_lower_bound': float(probs.max()) if complete else np.nan,
            'any_array_event_probability_upper_bound': float(min(1, probs.sum())) if complete else np.nan,
            'all_arrays_event_probability_lower_bound': float(max(0, probs.sum() - n + 1)) if complete else np.nan,
            'all_arrays_event_probability_upper_bound': float(probs.min()) if complete else np.nan,
            'complete_label_coverage': bool(known_labels),
            'observed_any_array_event_proxy': int(group[TARGET].max()) if known_labels else np.nan,
            'observed_all_arrays_event_proxy': int(group[TARGET].min()) if known_labels else np.nan,
            'signed_historical_profit_usd': float(group.historical_15yr_net_profit_usd.sum()),
            'current_capacity_tb': float(group.current_capacity_tb.sum()),
            'prediction_source': source,
            'semantics': 'array_event_summary_not_account_cancellation',
        })
    return pd.DataFrame(records)


def capacity_signals(raw):
    raw = validate_raw_frame(raw)
    frame = raw[KEYS + ['expansion_potential_signal', 'current_capacity_tb', 'utilization_rate_pct']].copy()
    frame['additional_capacity_to_reach_100pct_tb'] = (
        raw.current_capacity_tb * (raw.utilization_rate_pct - 1).clip(lower=0))
    frame['spare_capacity_at_reported_load_tb'] = (
        raw.current_capacity_tb * (1 - raw.utilization_rate_pct).clip(lower=0))
    frame['reported_over_capacity'] = raw.utilization_rate_pct > 1
    frame['expansion_signal_unknown'] = raw.expansion_potential_signal.eq('UNKNOWN')
    frame['signal_type'] = 'current_load_arithmetic_and_supplied_tag_not_future_prediction'
    return frame


def horizon_scenarios(raw, probabilities, cash_flows=None, discount_rates=(0, .05, .10)):
    """Compare 15-year constant-rate scenarios; optional keyed future cash flows."""
    raw = validate_raw_frame(raw)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(raw),):
        raise ValueError('One probability per array in snapshot order is required')
    survival_scenario(p)  # Validate even when later rate-loop is changed.
    rates = np.asarray(discount_rates, dtype=float)
    if rates.ndim != 1 or not len(rates) or not np.isfinite(rates).all() or (rates <= -1).any():
        raise ValueError('Discount rates must be finite and greater than -1')
    cash = np.ones((len(raw), 15))
    unit = 'one_unit_annual_cash_flow'
    if cash_flows is not None:
        flows = pd.read_csv(cash_flows, dtype=dict.fromkeys(KEYS, str)) if isinstance(cash_flows, (str, Path)) else cash_flows.copy()
        required = set(KEYS + ['year', 'annual_cash_flow_usd'])
        if required - set(flows) or flows.duplicated(KEYS + ['year']).any() or flows.isna().any().any():
            raise ValueError('Cash flows require unique populated account/array/year/cash-flow records')
        values = flows[['year', 'annual_cash_flow_usd']].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not flows.year.isin(range(1, 16)).all():
            raise ValueError('Provide finite cash flows and integer years 1 through 15')
        matrix = flows.pivot(index=KEYS, columns='year', values='annual_cash_flow_usd')
        index = pd.MultiIndex.from_frame(raw[KEYS])
        if set(matrix.index) != set(index) or set(matrix.columns) != set(range(1, 16)):
            raise ValueError('Cash flows must cover exactly the snapshot keys and all 15 years')
        cash = matrix.reindex(index=index, columns=range(1, 16)).to_numpy(dtype=float)
        if not np.isfinite(cash).all():
            raise ValueError('Every array must have all 15 annual cash flows')
        unit = 'usd_from_explicit_user_cash_flows'
    horizons, values = [], []
    for name, annual in [('calibrated_array_event_rate', p), ('static_5pct', np.full(len(raw), .05))]:
        survival = survival_scenario(annual)
        frame = raw[KEYS].copy()
        frame['scenario'] = name
        frame['assumed_annual_event_probability'] = annual
        for year in range(1, 16):
            frame[f'assumed_survival_year_{year}'] = survival[:, year - 1]
        frame['assumed_event_by_year_15'] = 1 - survival[:, -1]
        horizons.append(frame)
        for discount in rates:
            out = raw[KEYS].copy()
            out['scenario'] = name
            out['discount_rate'] = discount
            out['scenario_discounted_value'] = (cash * survival / (1 + discount) ** np.arange(1, 16)).sum(axis=1)
            out['value_units'] = unit
            values.append(out)
    return pd.concat(horizons, ignore_index=True), pd.concat(values, ignore_index=True)


def product_cohort_metrics(evidence, mapping):
    """Actual product-family lookup only; never infer families from other categories."""
    if 'prediction_source' not in evidence or not evidence.prediction_source.eq('grouped_oof').all():
        raise ValueError('Product-cohort evaluation requires grouped OOF evidence')
    mapping = pd.read_csv(mapping, dtype=dict.fromkeys(KEYS, str)) if isinstance(mapping, (str, Path)) else mapping.copy()
    if set(KEYS + ['product_family']) - set(mapping):
        raise ValueError('Product mapping requires account_id, array_id, product_family')
    if mapping.duplicated(KEYS).any() or mapping[KEYS + ['product_family']].isna().any().any():
        raise ValueError('Product mapping must be populated and unique')
    if mapping.product_family.astype(str).str.strip().eq('').any():
        raise ValueError('Product family must be populated')
    joined = evidence.merge(mapping[KEYS + ['product_family']], on=KEYS, how='left', validate='one_to_one')
    if joined.product_family.isna().any():
        raise ValueError('Product mapping does not cover all evaluated arrays')
    rows = []
    for family, group in joined.groupby('product_family'):
        for model in ['unweighted_xgb_sigmoid', 'unweighted_rf_sigmoid', PRIMARY_SCORE]:
            for baseline in ['static_5', 'training_prevalence']:
                rows.append({'product_family': family, 'model': model, 'baseline': baseline,
                             'arrays': len(group), 'accounts': group.account_id.nunique(),
                             **probability_metrics(group[TARGET].to_numpy(), group[model].to_numpy()),
                             **paired_account_ci(group, model, baseline)})
    return pd.DataFrame(rows)


def generate_submission_analytics(score_dir, run_dir=None, cash_flows=None, product_mapping=None):
    """Derived, immutable evidence tied to the actual scoring/training artifacts."""
    score_dir = Path(score_dir)
    score_manifest = json.loads((score_dir / 'score_manifest.json').read_text())
    bundle, manifest = load_bundle(run_dir)
    evidence, _ = read_evidence(run_dir)
    if score_manifest['training_run_id'] != manifest['run_id'] or score_manifest['bundle_sha256'] != manifest['bundle_sha256']:
        raise ValueError('Analytics scoring/training lineage mismatch')
    for file, field in [('input_snapshot.csv', 'snapshot_sha256'), ('CLV_SCORED_OUTPUT.csv', 'csv_sha256')]:
        if sha256_file(score_dir / file) != score_manifest[field]:
            raise ValueError('Analytics input hash mismatch')
    raw = validate_raw_frame(score_dir / 'input_snapshot.csv')
    training_snapshot = (Path(run_dir) if run_dir else latest_run()) / 'training_snapshot.csv'
    if sha256_file(training_snapshot) != manifest['training_snapshot_sha256']:
        raise ValueError('Training snapshot hash mismatch')
    probabilities = predict_bundle(bundle, raw)['ensemble']
    production = raw[KEYS].assign(event_probability=probabilities)
    horizons, values = horizon_scenarios(raw, probabilities, cash_flows)
    tables = {
        'account_array_event_summary.csv': account_summary(raw, production, 'event_probability', 'full_fit_scoring'),
        'account_oof_array_event_summary.csv': account_summary(validate_raw_frame(training_snapshot), evidence, PRIMARY_SCORE, 'grouped_oof'),
        'capacity_expansion_signals.csv': capacity_signals(raw),
        'fifteen_year_scenarios.csv': horizons,
        'discounted_cash_flow_scenarios.csv': values,
    }
    if product_mapping is not None:
        tables['product_family_cohorts.csv'] = product_cohort_metrics(evidence, product_mapping)
    root = score_dir / 'submission_analytics'
    root.mkdir(exist_ok=False)
    for file, table in tables.items():
        table.to_csv(root / file, index=False)
    metadata = {
        'training_run_id': manifest['run_id'], 'score_run_id': score_manifest['score_run_id'],
        'score_manifest_sha256': sha256_file(score_dir / 'score_manifest.json'),
        'evaluation_sha256': manifest['evaluation_sha256'],
        'artifacts': {file: {'rows': len(table), 'sha256': sha256_file(root / file)} for file, table in tables.items()},
        'cash_flow_source_sha256': input_fingerprint(cash_flows),
        'product_mapping_sha256': input_fingerprint(product_mapping),
        'account_semantics': 'Bounds on any/all array events; no account-cancellation prediction or independence assumption',
        'capacity_semantics': 'Current-load arithmetic with uncertain anomaly meaning; no learned future expansion',
        'horizon_semantics': 'Constant annual event-rate assumption; not learned survival or verified renewal',
        'cash_flow_semantics': 'Explicit supplied future USD cash flows, or one unit annually; historical profit is never extrapolated',
        'product_cohort_status': 'evaluated_from_supplied_mapping' if product_mapping is not None else 'unavailable_no_product_mapping',
    }
    (root / 'analytics_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return root, metadata
