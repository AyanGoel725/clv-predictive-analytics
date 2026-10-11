"""Validate the 25-column CSV contract before any scoring export is written."""
import numpy as np
import pandas as pd
from src.config import SCORE_COLUMNS
from src.data_cleaning import EXPANSION_SIGNAL_MAP
from src.utils import assign_risk_bucket


def validate_scored_output(df):
    if list(df.columns) != SCORE_COLUMNS or df.empty or df.isna().any().any():
        raise ValueError('Scoring output must have the exact populated 25-column contract')
    if df.duplicated(['account_id', 'array_id', 'model_version']).any():
        raise ValueError('Duplicate scoring keys')
    for c in ('account_id', 'array_id', 'model_version'):
        if df[c].astype(str).str.len().gt(50).any() or df[c].astype(str).str.strip().eq('').any():
            raise ValueError(f'{c} does not fit VARCHAR(50)')
    limits = {
        'historical_15yr_net_profit_usd': 10**13,
        'current_capacity_tb': 10**8,
        'utilization_rate_pct': 10**4,
        'raw_utilization_rate_pct': 10**4,
    }
    for c, bound in limits.items():
        if not np.isfinite(df[c]).all() or df[c].abs().ge(bound).any():
            raise ValueError(f'{c} exceeds warehouse precision')
    if (df.current_capacity_tb <= 0).any() or (df.raw_utilization_rate_pct < 0).any():
        raise ValueError('Exported capacity/utilization is invalid after rounding')
    if not np.allclose(df.utilization_rate_pct, df.raw_utilization_rate_pct, atol=1e-6, rtol=0):
        raise ValueError('Raw utilization columns disagree')
    if not np.allclose(df.utilization_capped, df.raw_utilization_rate_pct.clip(upper=1), atol=1e-6, rtol=0):
        raise ValueError('Capped utilization disagrees with raw utilization')
    if not df.is_over_utilized_raw.eq(df.raw_utilization_rate_pct > 1).all():
        raise ValueError('Over-utilization flag lost its meaning after export rounding')
    if not df.is_negative_profit.eq(df.historical_15yr_net_profit_usd < 0).all():
        raise ValueError('Negative-profit flag lost its meaning after export rounding')
    expected_signal = df.expansion_potential_signal.map(EXPANSION_SIGNAL_MAP)
    if expected_signal.isna().any() or not df.expansion_signal_encoded.eq(expected_signal).all():
        raise ValueError('Expansion signal/export code mismatch')
    if not np.allclose(df.static_baseline_churn_rate, .05, atol=1e-6, rtol=0):
        raise ValueError('Static baseline must be 5%')
    for c in ('xgb_churn_probability', 'rf_churn_probability', 'ensemble_churn_probability'):
        p = df[c].to_numpy()
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Exported probabilities must be finite in [0,1]')
    if not df.churn_risk_bucket.eq(df.ensemble_churn_probability.map(assign_risk_bucket)).all():
        raise ValueError('Exported buckets disagree with the probability policy')
    for i in range(1, 4):
        if not np.isfinite(df[f'shap_top_feature_{i}_value']).all():
            raise ValueError('Nonfinite SHAP values')
        if df[f'shap_top_feature_{i}'].str.len().gt(100).any():
            raise ValueError('SHAP feature name exceeds warehouse width')
    if pd.to_datetime(df.contract_end_date, errors='raise').isna().any():
        raise ValueError('Missing contract date')
    if pd.to_datetime(df.scored_at, errors='raise').isna().any():
        raise ValueError('Missing scoring timestamp')
    return True

