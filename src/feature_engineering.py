"""Fitted, serializable preprocessing with explicit feature order."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from src.config import REFERENCE_DATE
from src.data_cleaning import validate_raw_frame, SIGNALS


class CLVPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, reference_date=REFERENCE_DATE):
        self.reference_date = reference_date

    def fit(self, X, y=None):
        df = validate_raw_frame(X)
        cuts = np.unique(np.quantile(df.current_capacity_tb, [.25, .5, .75]))
        self.capacity_edges_ = np.r_[-np.inf, cuts, np.inf]
        self.feature_names_ = [
            'current_capacity_tb', 'is_over_utilized_raw', 'utilization_capped',
            'is_negative_profit', 'signed_log_profit', 'days_to_contract_end',
            'contract_year', 'contract_quarter', 'profit_per_tb', 'stress_index',
            'capacity_bin', *[f'expansion_{s.lower()}' for s in SIGNALS],
        ]
        self.fit_rows_ = len(df)
        return self

    def transform(self, X):
        check_is_fitted(self, 'capacity_edges_')
        df = validate_raw_frame(X)
        profit = df.historical_15yr_net_profit_usd.to_numpy()
        capped = df.utilization_rate_pct.clip(upper=1)
        neg = (profit < 0).astype(int)
        f = pd.DataFrame({
            'current_capacity_tb': df.current_capacity_tb,
            'is_over_utilized_raw': (df.utilization_rate_pct > 1).astype(int),
            'utilization_capped': capped, 'is_negative_profit': neg,
            'signed_log_profit': np.sign(profit) * np.log1p(np.abs(profit)),
            'days_to_contract_end': (df.contract_end_date - pd.Timestamp(self.reference_date)).dt.days.clip(lower=0),
            'contract_year': df.contract_end_date.dt.year,
            'contract_quarter': df.contract_end_date.dt.quarter,
            'profit_per_tb': profit / df.current_capacity_tb,
            'stress_index': capped * (1 + neg),
            'capacity_bin': np.searchsorted(self.capacity_edges_[1:-1], df.current_capacity_tb, side='left') + 1,
        })
        for s in SIGNALS:
            f[f'expansion_{s.lower()}'] = df.expansion_potential_signal.eq(s).astype(int)
        result = f[self.feature_names_].to_numpy(dtype=float)
        if not np.isfinite(result).all():
            raise ValueError('Transformed features must be finite')
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_')
        return np.asarray(self.feature_names_, dtype=object)


def build_feature_pipeline(df_labeled, df_unlabeled):
    """Exploration helper only; model CV fits its own fold-local transformer."""
    prep = CLVPreprocessor().fit(df_labeled)
    unseen = prep.transform(df_unlabeled) if len(df_unlabeled) else np.empty((0, len(prep.feature_names_)))
    return prep.transform(df_labeled), df_labeled['churn_event_next_12m'].to_numpy(), unseen, prep.feature_names_

