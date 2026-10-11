"""Explicit assumption sensitivity, never a fitted survival/CLV forecast."""
import numpy as np


def survival_scenario(annual_churn, years=15):
    p = np.asarray(annual_churn, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not isinstance(years, int) or years < 1:
        raise ValueError('Scenario requires finite p in [0,1] and positive years')
    return (1 - p[..., None]) ** np.arange(1, years + 1)


def discounted_clv_scenario(annual_cash_flows, annual_churn, discount_rate=.1):
    cash = np.asarray(annual_cash_flows, dtype=float)
    if cash.ndim != 1 or not len(cash) or not np.isfinite(cash).all() or not np.isfinite(discount_rate) or discount_rate <= -1:
        raise ValueError('Provide finite annual cash flows and discount_rate > -1')
    survival = survival_scenario(annual_churn, len(cash))
    return np.sum(survival * cash / (1 + discount_rate) ** np.arange(1, len(cash) + 1), axis=-1)

