"""Read immutable grouped OOF evidence; generate result-dependent evaluation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from src.config import OUTPUTS, PRIMARY_SCORE
from src.data_cleaning import TARGET
from src.model_training import latest_run, sha256_file
from src.utils import assign_risk_bucket


def score_columns(df):
    return [c for c in df if c in ('static_5', 'training_prevalence', 'logistic')
            or c.startswith(('weighted_', 'unweighted_'))]


def validate_evaluation(df, manifest):
    required = {'account_id', 'array_id', TARGET, 'fold_id', 'run_id', 'prediction_source'}
    if required - set(df):
        raise ValueError('Evaluation requires keyed OOF evidence, fold IDs and lineage')
    if manifest.get('prediction_source') != 'grouped_oof' or not df.prediction_source.eq('grouped_oof').all():
        raise ValueError('Fitted-production scores cannot be used for evaluation')
    if not df.run_id.eq(manifest['run_id']).all() or len(df) != manifest['labeled_rows']:
        raise ValueError('Evaluation run/row-count mismatch')
    if df.duplicated(['account_id', 'array_id']).any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError('Duplicate keys or invalid/missing evaluation labels')
    if df.groupby('account_id').fold_id.nunique().max() != 1:
        raise ValueError('Sibling arrays must stay in the same validation fold')
    all_accounts = set(df.account_id)
    ids = set()
    for fold in manifest['folds']:
        fid = fold['fold_id']
        if fid in ids:
            raise ValueError('Duplicate fold ID in manifest')
        ids.add(fid)
        tr, va = set(fold['train_accounts']), set(fold['validation_accounts'])
        actual = df.loc[df.fold_id.eq(fid)]
        if tr & va or tr != all_accounts - va or va != set(actual.account_id):
            raise ValueError('Training/validation account isolation failed')
        if len(actual) != fold['validation_rows']:
            raise ValueError('Fold row-count mismatch')
        if fold['train_rows'] != len(df) - len(actual):
            raise ValueError('Training row-count mismatch')
        expected_prevalence = df.loc[~df.fold_id.eq(fid), TARGET].mean()
        if not np.isclose(expected_prevalence, fold['training_prevalence']):
            raise ValueError('Prevalence baseline must use training labels only')
        if not np.allclose(actual.training_prevalence, fold['training_prevalence']):
            raise ValueError('Fold prevalence mismatch')
    if set(df.fold_id) != ids:
        raise ValueError('Unknown or missing fold assignment')
    cols = score_columns(df)
    if PRIMARY_SCORE not in cols or not {'static_5', 'training_prevalence', 'logistic'} <= set(cols):
        raise ValueError('Missing candidate or baseline probability columns')
    p = df[cols].to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Evaluation probabilities must be finite in [0, 1]')


def read_evidence(run_dir=None):
    run_dir = Path(run_dir) if run_dir is not None else latest_run()
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    file = run_dir / 'evaluation_predictions.csv'
    if sha256_file(file) != manifest['evaluation_sha256']:
        raise ValueError('OOF evidence hash mismatch')
    df = pd.read_csv(file, dtype={'account_id': str, 'array_id': str, 'run_id': str})
    validate_evaluation(df, manifest)
    return df, manifest


def expected_calibration_error(y, p, quantile=False):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11))) if quantile else np.linspace(0, 1, 11)
    bins = np.searchsorted(edges[1:-1], p, side='right')
    return float(sum(np.mean(bins == k) * abs(np.mean(p[bins == k]) - np.mean(y[bins == k]))
                     for k in np.unique(bins)))


def probability_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {
        'mean_probability': float(p.mean()), 'observed_rate': float(y.mean()),
        'bias': float(p.mean() - y.mean()), 'brier': float(np.mean((p - y) ** 2)),
        'log_loss': float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])),
        'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'average_precision': float(average_precision_score(y, p)) if y.sum() else None,
        'ece_equal_width': expected_calibration_error(y, p),
        'ece_equal_count': expected_calibration_error(y, p, quantile=True),
    }


def paired_account_ci(df, candidate, baseline, repetitions=1000, seed=20261011):
    """Paired fixed-prediction percentile CI; whole accounts are resampled."""
    y = df[TARGET].to_numpy()
    p, b = df[candidate].to_numpy(), df[baseline].to_numpy()
    values = pd.DataFrame({
        'account_id': df.account_id, 'n': 1,
        'gain': (b - y) ** 2 - (p - y) ** 2, 'bias': p - y,
    }).groupby('account_id').sum()
    rng = np.random.default_rng(seed)
    samples = rng.multinomial(len(values), np.full(len(values), 1 / len(values)), size=repetitions)
    n = samples @ values.n.to_numpy()
    gains = (samples @ values.gain.to_numpy()) / n
    biases = (samples @ values.bias.to_numpy()) / n
    return {
        'brier_gain': float(values.gain.sum() / values.n.sum()),
        'gain_ci_low': float(np.quantile(gains, .025)),
        'gain_ci_high': float(np.quantile(gains, .975)),
        'bias_ci_low': float(np.quantile(biases, .025)),
        'bias_ci_high': float(np.quantile(biases, .975)),
    }


def improvement_text(lift_pct):
    if abs(lift_pct) < 1e-10:
        return 'Brier score is unchanged relative to the baseline.'
    if lift_pct > 0:
        return f'Brier score improves by {lift_pct:.2f}% relative to the baseline.'
    return f'Brier score deteriorates by {-lift_pct:.2f}% relative to the baseline.'


def write_once(path, content):
    """Replaying a report preserves bytes; changed results require a new run."""
    path = Path(path)
    content = content.replace('\r\n', '\n')
    if path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise ValueError('Evaluation artifacts are immutable; use a new run for changed results')
        return
    path.write_text(content, encoding='utf-8')


def run_backtesting_suite(evaluation, manifest, output_dir, repetitions=1000):
    """Only validated keyed grouped-OOF evidence is accepted (retired array API removed)."""
    validate_evaluation(evaluation, manifest)
    df = evaluation.copy()
    y = df[TARGET].to_numpy()
    metrics = pd.DataFrame([
        {'model': c, **probability_metrics(y, df[c].to_numpy())} for c in score_columns(df)
    ])
    base_brier = float(metrics.loc[metrics.model.eq('static_5'), 'brier'].iloc[0])
    metrics['brier_lift_vs_5_pct'] = 100 * (base_brier - metrics.brier) / base_brier
    ci = pd.DataFrame([
        {'model': candidate, 'baseline': baseline,
         **paired_account_ci(df, candidate, baseline, repetitions=repetitions)}
        for candidate in ['weighted_ensemble_raw', 'unweighted_xgb_raw', 'unweighted_xgb_sigmoid',
                          'unweighted_rf_raw', 'unweighted_rf_sigmoid', 'unweighted_ensemble_raw',
                          PRIMARY_SCORE, 'training_prevalence', 'logistic']
        for baseline in ['static_5', 'training_prevalence']
    ])
    df['contract_year'] = pd.to_datetime(df.contract_end_date).dt.year
    cohorts = []
    for dimension in ('contract_year', 'expansion_potential_signal'):
        for cohort, group in df.groupby(dimension, dropna=False):
            cohorts.append({
                'dimension': dimension, 'cohort': str(cohort), 'n': len(group),
                **probability_metrics(group[TARGET].to_numpy(), group[PRIMARY_SCORE].to_numpy()),
                **paired_account_ci(group, PRIMARY_SCORE, 'static_5', repetitions=repetitions),
            })
    cohorts = pd.DataFrame(cohorts)
    df['risk_bucket'] = df[PRIMARY_SCORE].map(assign_risk_bucket)
    buckets = df.groupby('risk_bucket').agg(
        arrays=(TARGET, 'size'), churns=(TARGET, 'sum'), observed_rate=(TARGET, 'mean'),
        signed_historical_profit_usd=('historical_15yr_net_profit_usd', 'sum'),
    ).reindex(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'])
    buckets[['arrays', 'churns', 'signed_historical_profit_usd']] = buckets[
        ['arrays', 'churns', 'signed_historical_profit_usd']].fillna(0)
    if not np.isclose(buckets.signed_historical_profit_usd.sum(), df.historical_15yr_net_profit_usd.sum()):
        raise AssertionError('Historical exposure reconciliation failed')
    primary = metrics.set_index('model').loc[PRIMARY_SCORE]
    vs_prev = ci.loc[(ci.model == PRIMARY_SCORE) & (ci.baseline == 'training_prevalence')].iloc[0]
    conclusion = ('The paired CI supports a Brier gain over prevalence on these fixed predictions; '
                  'independent deployment validation is still needed.'
                  if vs_prev.gain_ci_low > 0 else
                  'No reliable Brier benefit over a fold-trained portfolio rate is established. '
                  'Retention targeting and a production champion claim are unsupported.')
    lines = [
        '# Grouped out-of-fold churn evaluation',
        '',
        f"Run: {manifest['run_id']} | Model: {manifest['model_version']}",
        f"Prediction source: grouped OOF | Labeled arrays: {len(df)} | Accounts: {df.account_id.nunique()}",
        f"Observed 12-month array-event rate: {y.mean():.4%}",
        '',
        'Outer folds keep accounts intact. Capacity bins fit within each training fold. '
        'Calibrators fit on inner account-grouped OOF scores; outer labels never fit the mapping.',
        'This is cross-sectional validation, not historical back-testing. Observation dates and label maturity are unverified.',
        '',
        '## Probability metrics',
        '',
        'Average precision is reported as AP, not trapezoidal PR area. Lower Brier/log loss is better. '
        'Positive bias means over-forecasting churn. Candidates and fixed weights were specified before this run; no outer-fold threshold tuning.',
        '',
        metrics.to_markdown(index=False, floatfmt='.6f'),
        '',
        f'Prespecified candidate: {PRIMARY_SCORE}.',
        improvement_text(primary.brier_lift_vs_5_pct),
        conclusion,
        '',
        '## Paired account bootstrap intervals',
        '',
        f'{repetitions} resamples; seed 20261011. Positive gain = lower Brier. '
        'Intervals condition on predictions and exclude refitting/model-selection uncertainty.',
        '',
        ci.to_markdown(index=False, floatfmt='.6f'),
        '',
        '## Diagnostic cohorts',
        '',
        'Contract expiry year and expansion tags are proxy segments, not product families or historical windows. '
        'Cohort variability does not prove feature usefulness.',
        '',
        cohorts.to_markdown(index=False, floatfmt='.6f'),
        '',
        '## Signed historical exposure by OOF risk bucket',
        '',
        'Policy thresholds: CRITICAL >=0.70; HIGH >=0.40; MEDIUM >=0.15; LOW <0.15. '
        'Buckets do not establish retention uplift. Empty-bucket observed rates are undefined.',
        '',
        buckets.to_markdown(floatfmt='.6f'),
        '',
        f'Signed labeled historical profit total: ${df.historical_15yr_net_profit_usd.sum():,.2f}. '
        'This is neither expected future loss nor discounted CLV.',
        '',
        '## Limitations and recommendation',
        '',
        'Array labels do not establish account cancellation. Future expansion outcomes, product families, '
        'observation timestamps, event durations and future cash flows are absent. '
        'A 15-year horizon requires explicit scenarios and cannot be claimed as observed renewal performance.',
        conclusion,
        'Live Snowflake procedure compilation/execution remains unverified.',
        '',
    ]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, table in [('metrics.csv', metrics), ('bootstrap_intervals.csv', ci), ('cohorts.csv', cohorts)]:
        write_once(root / name, table.to_csv(index=False))
    write_once(root / 'historical_exposure_buckets.csv', buckets.to_csv())
    report = '\n'.join(lines)
    write_once(root / 'backtesting_report.md', report)
    return {'metrics': metrics, 'intervals': ci, 'cohorts': cohorts, 'buckets': buckets, 'report': report}


if __name__ == '__main__':
    from src.pipeline import main
    main(['evaluate'])

