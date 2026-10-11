"""Canonical repository paths and exported scoring contract."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'Case_Study_2_CLV_Predictive_Analytics.csv'
OUTPUTS = ROOT / 'outputs'
MODEL_VERSION = 'v2.1.0-grouped-sigmoid-unweighted'
REFERENCE_DATE = '2026-10-01'
PRIMARY_SCORE = 'unweighted_ensemble_sigmoid'
SCORE_COLUMNS = [
    'account_id', 'array_id', 'contract_end_date', 'historical_15yr_net_profit_usd',
    'current_capacity_tb', 'utilization_rate_pct', 'raw_utilization_rate_pct',
    'utilization_capped', 'is_over_utilized_raw', 'is_negative_profit',
    'expansion_potential_signal', 'expansion_signal_encoded',
    'static_baseline_churn_rate', 'xgb_churn_probability', 'rf_churn_probability',
    'ensemble_churn_probability', 'churn_risk_bucket', 'shap_top_feature_1',
    'shap_top_feature_1_value', 'shap_top_feature_2', 'shap_top_feature_2_value',
    'shap_top_feature_3', 'shap_top_feature_3_value', 'scored_at', 'model_version',
]

