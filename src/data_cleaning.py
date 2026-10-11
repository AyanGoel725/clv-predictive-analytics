"""Validate snapshots; preserve anomalies without learning batch statistics."""
from pathlib import Path
import numpy as np
import pandas as pd
TARGET = 'churn_event_next_12m'
SIGNALS = ('UNKNOWN', 'Low', 'Medium', 'High')
EXPANSION_SIGNAL_MAP = dict(zip(SIGNALS, range(4)))  # Export code, not an ML ordering.
REQUIRED = ('account_id', 'array_id', 'contract_end_date',
            'historical_15yr_net_profit_usd', 'current_capacity_tb',
            'utilization_rate_pct', 'expansion_potential_signal')


def validate_raw_frame(data, require_target=False):
    if not isinstance(data, (str, Path, pd.DataFrame)):
        raise ValueError('Provide a raw DataFrame or CSV path, not precomputed feature arrays')
    df = pd.read_csv(data, dtype={'account_id': str, 'array_id': str}) if isinstance(data, (str, Path)) else data.copy()
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df.empty:
        raise ValueError('Snapshot must contain at least one row')
    for col in ('account_id', 'array_id'):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'{col} must be populated')
        df[col] = df[col].astype(str)
    if df.duplicated(['account_id', 'array_id']).any():
        raise ValueError('Duplicate account/array keys')
    for col in ('historical_15yr_net_profit_usd', 'current_capacity_tb', 'utilization_rate_pct'):
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(df[col]).all():
            raise ValueError(f'{col} must be finite and populated')
    if (df.current_capacity_tb <= 0).any():
        raise ValueError('Capacity must be positive')
    if (df.utilization_rate_pct < 0).any():
        raise ValueError('Utilization must be nonnegative (fraction units)')
    df['contract_end_date'] = pd.to_datetime(df.contract_end_date, errors='raise')
    if df.contract_end_date.isna().any():
        raise ValueError('Contract dates must be populated')
    df['expansion_potential_signal'] = df.expansion_potential_signal.fillna('UNKNOWN')
    if not df.expansion_potential_signal.isin(SIGNALS).all():
        raise ValueError('Unsupported expansion category; supported: UNKNOWN/Low/Medium/High')
    if TARGET not in df:
        if require_target:
            raise ValueError('Training requires the churn target column')
        df[TARGET] = np.nan
    df[TARGET] = pd.to_numeric(df[TARGET], errors='raise')
    if not df[TARGET].dropna().isin([0, 1]).all():
        raise ValueError('Churn target must be 0, 1, or missing')
    return df.reset_index(drop=True)


def audit_anomalies(df):
    audit = {'total_rows': len(df)}
    for name, mask in {
        'over_utilized': df.utilization_rate_pct > 1,
        'negative_profit': df.historical_15yr_net_profit_usd < 0,
        'missing_churn': df[TARGET].isna(),
        'unknown_expansion': df.expansion_potential_signal.eq('UNKNOWN'),
    }.items():
        audit[f'{name}_count'] = int(mask.sum())
        audit[f'{name}_pct'] = float(mask.mean() * 100)
    audit.update(max_raw_utilization=float(df.utilization_rate_pct.max()),
                 min_profit_usd=float(df.historical_15yr_net_profit_usd.min()))
    return audit


def clean_dataset(filepath_or_df, reference_min_profit=None):
    """Signed-log cleaning uses no fitted offset; targets are never imputed."""
    if reference_min_profit is not None:
        raise ValueError('v2 uses signed log; shifted-profit offsets are no longer supported')
    df = validate_raw_frame(filepath_or_df)
    audit = audit_anomalies(df)
    df['raw_utilization_rate_pct'] = df.utilization_rate_pct
    df['utilization_capped'] = df.utilization_rate_pct.clip(upper=1)
    df['is_over_utilized_raw'] = (df.utilization_rate_pct > 1).astype(int)
    df['is_negative_profit'] = (df.historical_15yr_net_profit_usd < 0).astype(int)
    df['signed_log_profit'] = np.sign(df.historical_15yr_net_profit_usd) * np.log1p(df.historical_15yr_net_profit_usd.abs())
    df['expansion_signal_encoded'] = df.expansion_potential_signal.map(EXPANSION_SIGNAL_MAP)
    labeled = df.loc[df[TARGET].notna()].reset_index(drop=True)
    unlabeled = df.loc[df[TARGET].isna()].reset_index(drop=True)
    labeled[TARGET] = labeled[TARGET].astype(int)
    audit.update(labeled_rows=len(labeled), unlabeled_rows=len(unlabeled),
                 labeled_churn_count=int(labeled[TARGET].sum()),
                 labeled_churn_rate_pct=float(labeled[TARGET].mean() * 100))
    return labeled, unlabeled, audit


if __name__ == '__main__':
    from src.config import DATA_PATH
    print(clean_dataset(DATA_PATH)[2])

