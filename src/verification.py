"""Read-only delivery checks; does not train, score, or generate reports."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import ROOT, DATA_PATH, OUTPUTS, SCORE_COLUMNS
from src.data_cleaning import clean_dataset, validate_raw_frame, TARGET
from src.feature_engineering import CLVPreprocessor
from src.model_training import latest_run, load_bundle, frame_hash, predict_bundle, sha256_file
from src.backtesting import read_evidence
from src.utils import assign_risk_bucket
from src.contracts import validate_scored_output


def verify():
    run_dir = latest_run()
    evidence, manifest = read_evidence(run_dir)
    bundle, _ = load_bundle(run_dir)
    assert set(manifest['source_sha256']) == {p.name for p in (ROOT / 'src').glob('*.py')}
    assert set(manifest['warehouse_source_sha256']) == {p.name for p in (ROOT / 'sql').glob('*.sql')}
    for name, expected_hash in manifest.get('source_sha256', {}).items():
        assert sha256_file(ROOT / 'src' / name) == expected_hash, f'Source changed since training: {name}'
    for name, expected_hash in manifest['warehouse_source_sha256'].items():
        assert sha256_file(ROOT / 'sql' / name) == expected_hash, f'SQL changed since training: {name}'
    assert sha256_file(run_dir / 'training_snapshot.csv') == manifest['training_snapshot_sha256']
    raw = validate_raw_frame(DATA_PATH)
    labeled, unlabeled, audit = clean_dataset(raw)
    assert frame_hash(validate_raw_frame(labeled)) == manifest['labeled_input_sha256'], 'Training input hash mismatch'
    assert len(labeled) == len(evidence)
    assert audit['labeled_churn_count'] == int(evidence[TARGET].sum())
    assert np.array_equal(evidence[['account_id', 'array_id']].to_numpy(),
                          labeled[['account_id', 'array_id']].to_numpy())
    assert np.array_equal(evidence[TARGET].to_numpy(), labeled[TARGET].to_numpy())
    new = unlabeled.drop(columns=TARGET)
    assert np.isfinite(bundle['preprocessor'].transform(new)).all(), 'All-unlabeled inference failed'
    pointer = json.loads((OUTPUTS / 'latest_score_run.json').read_text())
    score_dir = OUTPUTS / pointer['directory']
    score_manifest = json.loads((score_dir / 'score_manifest.json').read_text())
    for name, field in [('CLV_SCORED_OUTPUT.csv', 'csv_sha256'), ('input_snapshot.csv', 'snapshot_sha256'),
                        ('xgb_explanations.json', 'explanations_sha256')]:
        assert sha256_file(score_dir / name) == score_manifest[field], f'{name} hash mismatch'
    scored = pd.read_csv(score_dir / 'CLV_SCORED_OUTPUT.csv', dtype={'account_id': str, 'array_id': str})
    validate_scored_output(scored)
    snapshot = validate_raw_frame(score_dir / 'input_snapshot.csv')
    assert score_manifest['training_run_id'] == manifest['run_id']
    assert list(scored.columns) == SCORE_COLUMNS and not scored.isna().any().any()
    assert len(scored) == len(snapshot) == score_manifest['rows']
    assert not scored.duplicated(['account_id', 'array_id', 'model_version']).any()
    assert scored.churn_risk_bucket.tolist() == scored.ensemble_churn_probability.map(assign_risk_bucket).tolist()
    p = predict_bundle(bundle, snapshot)
    for name in ('xgb', 'rf', 'ensemble'):
        np.testing.assert_allclose(scored[f'{name}_churn_probability'], p[name], atol=5.1e-7)
    analytics_dir = score_dir / 'submission_analytics'
    analytics = json.loads((analytics_dir / 'analytics_manifest.json').read_text())
    assert analytics['training_run_id'] == manifest['run_id']
    assert analytics['score_run_id'] == score_manifest['score_run_id']
    assert analytics['evaluation_sha256'] == manifest['evaluation_sha256']
    assert analytics['score_manifest_sha256'] == sha256_file(score_dir / 'score_manifest.json')
    for filename, metadata in analytics['artifacts'].items():
        assert sha256_file(analytics_dir / filename) == metadata['sha256'], filename
        assert len(pd.read_csv(analytics_dir / filename)) == metadata['rows'], filename
    accounts = pd.read_csv(analytics_dir / 'account_array_event_summary.csv')
    assert len(accounts) == snapshot.account_id.nunique()
    assert accounts.complete_probability_coverage.all()
    assert np.isclose(accounts.signed_historical_profit_usd.sum(), snapshot.historical_15yr_net_profit_usd.sum())
    horizons = pd.read_csv(analytics_dir / 'fifteen_year_scenarios.csv')
    assert len(horizons) == 2 * len(snapshot)
    for name in ['calibrated_array_event_rate', 'static_5pct']:
        group = horizons.loc[horizons.scenario.eq(name)]
        for year in range(1, 16):
            np.testing.assert_allclose(group[f'assumed_survival_year_{year}'],
                                       (1 - group.assumed_annual_event_probability)**year, atol=1e-14)
    report = run_dir / 'backtesting_report.md'
    assert report.exists() and 'grouped OOF' in report.read_text()
    assert sha256_file(report) == sha256_file(OUTPUTS / 'backtesting_report.md')
    metrics = pd.read_csv(run_dir / 'metrics.csv').set_index('model')
    y = evidence[TARGET].to_numpy()
    for model in metrics.index:
        assert np.isclose(metrics.loc[model, 'brier'], np.mean((evidence[model].to_numpy() - y)**2)), model
    ddl = (ROOT / 'sql/ddl_clv_scored_output.sql').read_text(encoding='utf-8-sig')
    body = ddl.split('CREATE TABLE IF NOT EXISTS CLV_SCORE_STAGING_TEMPLATE (')[1].split(');')[0]
    assert [line.strip().split()[0].lower() for line in body.splitlines() if line.strip()] == SCORE_COLUMNS
    assert 'CREATE TABLE IF NOT EXISTS CLV_SCORED_OUTPUT LIKE CLV_SCORED_HISTORY' in ddl
    assert 'CREATE OR REPLACE VIEW CLV_SCORED_OUTPUT' not in ddl
    handler = (ROOT / 'sql/stored_proc_scoring_pipeline.sql').read_text(encoding='utf-8-sig').split('$$')[1]
    compile(handler, 'snowflake_handler', 'exec')
    assert 'FROM @CLV_STAGE' in handler and 'TRUNCATE' not in handler
    assert 'MERGE INTO CLV_SCORED_OUTPUT' in handler and 'LATEST_MERGE_QUERY_ID' in handler
    for path in (ROOT / 'notebooks').glob('*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        assert not [o for c in notebook['cells'] for o in c.get('outputs', []) if o['output_type'] == 'error'], path
        assert all(c['execution_count'] is not None for c in notebook['cells'] if c['cell_type'] == 'code'), path
    print(f'PASS: input/evidence/bundle/output hashes, grouped folds, unlabeled inference, probabilities, '
          f'OOF metrics, account summaries, 15-year scenarios, CSV schema and executed notebooks. Run {manifest["run_id"]}')
    print('LIMITATION: SQL/Python static checks only; live Snowflake execution remains unverified.')
    return True

