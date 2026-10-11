"""Transform snapshots with saved state; keep production scores out of evaluation."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import DATA_PATH, OUTPUTS, SCORE_COLUMNS
from src.data_cleaning import validate_raw_frame, EXPANSION_SIGNAL_MAP
from src.model_training import load_bundle, predict_bundle, sha256_file
from src.explainability import xgb_contributions, extract_top_shap_features
from src.utils import assign_risk_bucket
from src.contracts import validate_scored_output


def generate_production_scores(csv_path=DATA_PATH, run_dir=None, output_path=None):
    raw = validate_raw_frame(csv_path)
    bundle, training_manifest = load_bundle(run_dir)
    pred = predict_bundle(bundle, raw)
    values, base, _ = xgb_contributions(bundle, raw)
    names = bundle['preprocessor'].feature_names_
    top = extract_top_shap_features(values, names)
    score_run_id = uuid.uuid4().hex
    scored_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    df = pd.DataFrame({
        'account_id': raw.account_id, 'array_id': raw.array_id,
        'contract_end_date': raw.contract_end_date.dt.strftime('%Y-%m-%d'),
        'historical_15yr_net_profit_usd': raw.historical_15yr_net_profit_usd.round(2),
        'current_capacity_tb': raw.current_capacity_tb.round(2),
        'utilization_rate_pct': raw.utilization_rate_pct.round(6),
        'raw_utilization_rate_pct': raw.utilization_rate_pct.round(6),
        'utilization_capped': raw.utilization_rate_pct.clip(upper=1).round(6),
        'is_over_utilized_raw': raw.utilization_rate_pct > 1,
        'is_negative_profit': raw.historical_15yr_net_profit_usd < 0,
        'expansion_potential_signal': raw.expansion_potential_signal,
        'expansion_signal_encoded': raw.expansion_potential_signal.map(EXPANSION_SIGNAL_MAP),
        'static_baseline_churn_rate': .05,
        'xgb_churn_probability': np.round(pred['xgb'], 6),
        'rf_churn_probability': np.round(pred['rf'], 6),
        'ensemble_churn_probability': np.round(pred['ensemble'], 6),
        'churn_risk_bucket': [assign_risk_bucket(p) for p in np.round(pred['ensemble'], 6)],
        'scored_at': scored_at, 'model_version': bundle['model_version'],
    })
    for col in top:
        df[col] = top[col].round(6) if col.endswith('_value') else top[col]
    df = df[SCORE_COLUMNS]
    validate_scored_output(df)
    output_path = Path(output_path) if output_path else OUTPUTS / 'score_runs' / score_run_id / 'CLV_SCORED_OUTPUT.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError('Scoring runs are immutable; choose a fresh output path')
    df.to_csv(output_path, index=False)
    explanations = {
        'score_run_id': score_run_id, 'model_version': bundle['model_version'],
        'explained_model': 'unweighted_xgb_raw', 'units': 'raw_margin_log_odds',
        'explanation_scope': 'XGB component only, not calibrated ensemble probability or causal effects',
        'rows': [
            {'account_id': a, 'array_id': b, 'base_value': float(v),
             'contributions': dict(zip(names, map(float, row)))}
            for a, b, v, row in zip(raw.account_id, raw.array_id, base, values)
        ],
    }
    explanation_path = output_path.with_name('xgb_explanations.json')
    explanation_path.write_text(json.dumps(explanations, indent=2), encoding='utf-8')
    raw_path = output_path.with_name('input_snapshot.csv')
    raw.to_csv(raw_path, index=False, date_format='%Y-%m-%d')
    metadata = {
        'score_run_id': score_run_id, 'training_run_id': bundle['training_run_id'],
        'model_version': bundle['model_version'], 'scored_at_utc': scored_at,
        'reference_date': bundle['reference_date'], 'target_horizon': 'next_12_months_array_event',
        'snapshot_sha256': sha256_file(raw_path), 'bundle_sha256': training_manifest['bundle_sha256'],
        'original_input_sha256': sha256_file(csv_path) if isinstance(csv_path, (str, Path)) else None,
        'csv_sha256': sha256_file(output_path), 'explanations_sha256': sha256_file(explanation_path),
        'columns': SCORE_COLUMNS, 'rows': len(df), 'shap_units': explanations['units'],
        'primary_score': bundle['primary_score'],
        'probability_semantics': {
            'xgb': 'sigmoid mapping of raw XGB probability',
            'rf': 'sigmoid mapping of raw RF probability',
            'ensemble': 'sigmoid mapping of 0.6*raw_XGB + 0.4*raw_RF; not the weighted sum of mapped components',
        },
        'policy_thresholds': {'CRITICAL': .70, 'HIGH': .40, 'MEDIUM': .15, 'LOW': 0},
        'deployment_status': 'candidate; live Snowflake execution unverified',
    }
    output_path.with_name('score_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return df, metadata, output_path


if __name__ == '__main__':
    from src.pipeline import main
    main(['score'])

