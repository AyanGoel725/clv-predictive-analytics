"""Explain the XGB component in raw-margin/log-odds units."""
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb


def xgb_contributions(bundle, raw):
    X = bundle['preprocessor'].transform(raw)
    booster = bundle['xgb'].get_booster()
    values = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    margin = booster.predict(xgb.DMatrix(X), output_margin=True)
    if not np.allclose(values.sum(axis=1), margin, atol=1e-5):
        raise AssertionError('SHAP contributions fail raw-margin additivity')
    prob = 1 / (1 + np.exp(-margin))
    if not np.allclose(prob, bundle['xgb'].predict_proba(X)[:, 1], atol=1e-6):
        raise AssertionError('XGB raw margin does not reconstruct raw probability')
    return values[:, :-1], values[:, -1], X


def extract_top_shap_features(shap_values, feature_names, top_n=3):
    records = []
    for values in shap_values:
        indices = np.argsort(np.abs(values))[::-1][:top_n]
        records.append({item: value for rank, index in enumerate(indices, 1)
                        for item, value in (
                            (f'shap_top_feature_{rank}', feature_names[index]),
                            (f'shap_top_feature_{rank}_value', float(values[index])),
                        )})
    return pd.DataFrame(records)


def generate_shap_analysis(bundle, raw, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shap
    values, base, X = xgb_contributions(bundle, raw)
    names = bundle['preprocessor'].feature_names_
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind, filename in [('bar', 'shap_summary_bar.png'), ('dot', 'shap_beeswarm.png')]:
        shap.summary_plot(values, X, feature_names=names, plot_type=kind, show=False)
        plt.title('XGB component contributions in raw margin (log-odds)')
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close('all')
    for name, filename in [('utilization_capped', 'shap_dependence_utilization.png'),
                           ('signed_log_profit', 'shap_dependence_profit.png')]:
        shap.dependence_plot(names.index(name), values, X, feature_names=names,
                             interaction_index=None, show=False)
        plt.ylabel('XGB contribution (log-odds)')
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close('all')
    return values, base

