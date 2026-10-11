"""Build notebooks that inspect saved evidence without retraining."""
from pathlib import Path
import nbformat as nbf
ROOT = Path(__file__).resolve().parents[1]
SETUP = """from pathlib import Path
import sys
import json
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/config.py').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import DATA_PATH, OUTPUTS, PRIMARY_SCORE
from src.data_cleaning import validate_raw_frame, clean_dataset, TARGET
from src.model_training import latest_run, load_bundle, predict_bundle, sha256_file, frame_hash
from src.backtesting import read_evidence
plt.rcParams.update({'figure.figsize': (9, 4), 'figure.dpi': 110})
run_dir = latest_run()
oof, manifest = read_evidence(run_dir)
bundle, _ = load_bundle(run_dir)
for name, expected in manifest['source_sha256'].items():
    assert sha256_file(ROOT / 'src' / name) == expected, f'Source changed since training: {name}'
for name, expected in manifest['warehouse_source_sha256'].items():
    assert sha256_file(ROOT / 'sql' / name) == expected, f'SQL changed since training: {name}'
assert sha256_file(run_dir / 'training_snapshot.csv') == manifest['training_snapshot_sha256']
display(pd.Series({
    'training_run': manifest['run_id'], 'model_version': manifest['model_version'],
    'training_created_at_utc': manifest['created_at_utc'],
    'feature_reference_date': bundle['reference_date'],
    'validation_source': manifest['prediction_source'],
}).to_frame('saved evidence'))
print('Notebook review updated: 2026-10-11. Results come from the saved training run above.')
"""


def notebook(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(text) if kind == 'md' else nbf.v4.new_code_cell(text)
                for kind, text in cells]
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nbf.write(nb, ROOT / 'notebooks' / name)


def main():
    notebook('01_EDA_and_Data_Cleaning.ipynb', [
        ('md', '# Data audit and target limitations\n\nThe supplied target is a next-12-month array event. Missing labels are inference rows. Account cancellation, observation timestamps and label maturity are unverified. Anomaly causes are unknown.'),
        ('code', SETUP),
        ('code', """raw = validate_raw_frame(DATA_PATH)
labeled, unlabeled, audit = clean_dataset(raw)
display(pd.Series(audit, name='value').to_frame())
display(raw.describe(include='all').T)
assert len(labeled) + len(unlabeled) == len(raw)
assert set(labeled[TARGET]) == {0, 1}
assert frame_hash(validate_raw_frame(labeled)) == manifest['labeled_input_sha256']
"""),
        ('md', '## Label grain\n\nSibling arrays may have discordant labels. Neither one array retiring nor both arrays leaving proves commercial account loss.'),
        ('code', """accounts = labeled.groupby('account_id')[TARGET].agg(['count', 'nunique'])
fully_labeled = accounts[accounts['count'] == 2]
print('Fully labeled two-array accounts:', len(fully_labeled))
print('Discordant-label accounts:', int((fully_labeled['nunique'] > 1).sum()))
print('Unlabeled rows are excluded from supervised fitting:', len(unlabeled))
"""),
        ('md', '## Preserved anomalies\n\nUtilization is a fraction, so 1.45 means 145%. Profit remains signed; signed log avoids fitting a global minimum. UNKNOWN is nominal. Schema validation rejects unsupported categories, dates, labels, invalid capacity and duplicate keys.'),
        ('code', """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(raw.utilization_rate_pct, bins=25, color='#245b8a')
axes[0].axvline(1, color='#c45a36', linestyle='--', label='100%')
axes[0].set(xlabel='Raw utilization (fraction)', ylabel='Arrays', title='Utilization audit')
axes[0].legend()
axes[1].hist(labeled.signed_log_profit, bins=25, color='#34856b')
axes[1].set(xlabel='sign(profit) * log1p(abs(profit))', ylabel='Arrays', title='Signed profit transform')
plt.tight_layout()
plt.show()
display(raw.groupby('expansion_potential_signal', dropna=False).size().rename('arrays').to_frame())
"""),
        ('md', '## Cleaning and input contract\n\nAccount/array identifiers retain leading zeros when read from CSV. Dates and finite numeric values are required; capacity must be positive and utilization nonnegative. Labels can be 0, 1 or absent. Missing expansion tags become UNKNOWN, and unsupported values fail validation. Utilization above 100% and negative profit remain visible through explicit flags. The integer expansion export code is excluded from model features; four nominal indicators are used instead. No target, identifier or historical batch minimum enters the feature matrix.'),
        ('code', """anomaly_examples = labeled.loc[
    labeled.is_over_utilized_raw.eq(1) | labeled.is_negative_profit.eq(1),
    ['account_id', 'array_id', 'raw_utilization_rate_pct', 'utilization_capped',
     'is_over_utilized_raw', 'historical_15yr_net_profit_usd', 'signed_log_profit',
     'is_negative_profit', 'expansion_potential_signal']
]
display(anomaly_examples.head(8))
assert labeled.utilization_capped.le(1).all()
assert np.allclose(labeled.signed_log_profit,
                   np.sign(labeled.historical_15yr_net_profit_usd) *
                   np.log1p(labeled.historical_15yr_net_profit_usd.abs()))
print('Repeated arrays per account are handled by grouping folds, not by merging their labels.')
"""),
        ('md', '## Financial quantities\n\nThese are sums of signed historical profit. They are not expected losses or future CLV.'),
        ('code', """display(pd.DataFrame({
    'population': ['all arrays', 'labeled arrays'],
    'arrays': [len(raw), len(labeled)],
    'signed_historical_profit_usd': [raw.historical_15yr_net_profit_usd.sum(),
                                   labeled.historical_15yr_net_profit_usd.sum()]
}))
"""),
        ('md', '## Data still needed\n\nThe fixes do not establish event definitions or label maturity. Account cancellation rules, observation dates, product-family mappings, dated expansion outcomes, event/censoring histories and forward cash flows remain required for the corresponding business analyses. The frozen feature reference date shown above is a training setting; updating a notebook does not advance it.'),
    ])
    notebook('02_Feature_Engineering_and_Modeling.ipynb', [
        ('md', '# Grouped validation, probability calibration and component explanations\n\nRun python -m src.pipeline run first. This notebook reads saved evidence and does not retrain, tune thresholds, or overwrite reports.'),
        ('code', SETUP),
        ('code', """display(pd.Series({k: manifest[k] for k in ['run_id', 'validation_design', 'outer_folds',
                                          'inner_folds', 'primary_score', 'candidate_policy']}).to_frame('value'))
for fold in manifest['folds']:
    assert not set(fold['train_accounts']) & set(fold['validation_accounts'])
display(pd.DataFrame({'feature': bundle['preprocessor'].feature_names_}))
"""),
        ('md', '## Fold isolation and persisted feature state\n\nFor each outer split: fit preprocessing and models using training accounts; generate inner grouped OOF probabilities within those training accounts; fit sigmoid/isotonic maps using those inner predictions; then score untouched outer accounts. Fit the final inference bundle on all labeled rows with final maps learned from grouped inner OOF predictions. Capacity quartiles are learned separately inside each training fold and saved for inference. Signed log and nominal expansion indicators require no population-wide fitted offset.'),
        ('code', """display(pd.DataFrame([
    {'fold': f['fold_id'], 'train_accounts': len(f['train_accounts']),
     'validation_accounts': len(f['validation_accounts']),
     'train_rows': f['train_rows'], 'validation_rows': f['validation_rows'],
     'train_event_rate': f['training_prevalence'], 'capacity_edges': f['capacity_edges']}
    for f in manifest['folds']
]))
assert oof.groupby('account_id').fold_id.nunique().eq(1).all()
assert not oof.duplicated(['account_id', 'array_id']).any()
assert sum(f['validation_rows'] for f in manifest['folds']) == len(oof)
display(pd.Series({
    'saved_preprocessor_fit_rows': bundle['preprocessor'].fit_rows_,
    'saved_capacity_edges': bundle['preprocessor'].capacity_edges_.tolist(),
    'seed': manifest['seed'], 'tree_counts': manifest['parameters'],
    'calibration_maps': {name: mapping.method for name, mapping in bundle['calibrators'].items()},
}).to_frame('value'))
display(pd.DataFrame([
    {'package': name, 'training_version': version,
     'notebook_version': importlib.metadata.version(name)}
    for name, version in manifest['versions'].items()
]))
"""),
        ('md', '## Proper scores and baselines\n\nPreprocessing fits inside every training fold. Sigmoid/isotonic mappings fit to inner grouped OOF probabilities. Ensemble weights are fixed at 0.6/0.4. Outer labels do not select the scoring candidate or action thresholds. AP means average precision.'),
        ('code', """metrics = pd.read_csv(run_dir / 'metrics.csv').set_index('model')
display(metrics)
comparisons = ['static_5', 'training_prevalence', 'weighted_ensemble_raw',
               'unweighted_ensemble_raw', PRIMARY_SCORE]
metrics.loc[comparisons, 'brier'].plot.barh(color='#245b8a')
plt.xlabel('Out-of-fold Brier score (lower is better)')
plt.title('Probability score versus simple baselines')
plt.tight_layout()
plt.show()
"""),
        ('md', '## XGBoost versus Random Forest\n\nThe individual models are evaluated on the same account-isolated validation rows. Raw and sigmoid results show the effects of probability mapping. The prespecified ensemble remains an evaluated candidate; individual results are not used to select a new winner on these outer folds.'),
        ('code', """individuals = ['static_5', 'training_prevalence', 'unweighted_xgb_raw',
               'unweighted_rf_raw', 'unweighted_xgb_sigmoid', 'unweighted_rf_sigmoid', PRIMARY_SCORE]
display(metrics.loc[individuals, ['mean_probability', 'bias', 'brier', 'log_loss',
                                 'roc_auc', 'average_precision', 'ece_equal_count']])
individual_ci = pd.read_csv(run_dir / 'bootstrap_intervals.csv')
display(individual_ci.loc[individual_ci.model.isin(individuals)])
"""),
        ('md', 'Weighted candidates use fold-training class weights; unweighted candidates remove them. The logistic challenger fits a scaler and regularized logistic model within each outer fold. The prevalence challenger uses only that fold\'s training labels. Sigmoid maps fit a logistic mapping of clipped probability logits; isotonic maps fit monotonic probability mappings. Calibration changes probabilities; changing a classification threshold would not calibrate them. Risk buckets remain prespecified at 0.15, 0.40 and 0.70.'),
        ('code', """from sklearn.calibration import calibration_curve
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name in ['weighted_ensemble_raw', 'unweighted_ensemble_raw', PRIMARY_SCORE]:
    for ax, strategy in zip(axes, ['uniform', 'quantile']):
        observed, forecast = calibration_curve(oof[TARGET], oof[name], n_bins=10, strategy=strategy)
        ax.plot(forecast, observed, marker='o', label=name)
for ax, title in zip(axes, ['Equal-width bins', 'Equal-count bins']):
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='Mean forecast', ylabel='Observed array-event rate', title=title, xlim=(0, 1), ylim=(0, 1))
axes[0].legend(fontsize=7)
plt.tight_layout()
plt.show()
print('Small equal-width ECE alone does not demonstrate useful ranking.')
"""),
        ('md', '## Saved-state inference and honest SHAP units\n\nThis example scores an unlabeled array using saved state. SHAP explains only the raw XGB component in log-odds, not the mapped ensemble probability. It is associational, not causal.'),
        ('code', """from src.explainability import xgb_contributions, extract_top_shap_features
raw = validate_raw_frame(DATA_PATH)
example = raw.loc[raw[TARGET].isna()].iloc[:1].drop(columns=TARGET)
prediction = predict_bundle(bundle, example)
values, base, X = xgb_contributions(bundle, example)
display(example[['account_id', 'array_id']])
print('Mapped ensemble probability:', float(prediction['ensemble'][0]))
print('XGB raw margin:', float(values[0].sum() + base[0]))
display(extract_top_shap_features(values, bundle['preprocessor'].feature_names_))
all_unlabeled = raw.loc[raw[TARGET].isna()].drop(columns=TARGET)
saved_edges = bundle['preprocessor'].capacity_edges_.copy()
all_predictions = predict_bundle(bundle, all_unlabeled)
missing_target_predictions = predict_bundle(bundle, all_unlabeled.assign(**{TARGET: np.nan}))
for name in ['xgb', 'rf', 'ensemble']:
    assert np.isfinite(all_predictions[name]).all()
    np.testing.assert_allclose(all_predictions[name], missing_target_predictions[name])
np.testing.assert_array_equal(saved_edges, bundle['preprocessor'].capacity_edges_)
print('All unlabeled rows scored without refitting:', len(all_unlabeled))
raw_xgb_p = bundle['xgb'].predict_proba(X)[:, 1]
np.testing.assert_allclose(1 / (1 + np.exp(-(values.sum(axis=1) + base))), raw_xgb_p, atol=1e-6)
"""),
        ('md', '## Global component explanations\n\nThe four plots use the full supplied snapshot and the final fitted XGB component. They illustrate model associations and are not validation evidence. Full per-row contributions and base values are retained in the scoring sidecar; the CSV keeps the top three display features.'),
        ('code', """for filename in ['shap_summary_bar.png', 'shap_beeswarm.png',
                 'shap_dependence_utilization.png', 'shap_dependence_profit.png']:
    display(Image(filename=str(OUTPUTS / 'shap_plots' / filename)))
"""),
        ('md', 'The prespecified candidate is an evaluated option. A learned portfolio rate can match or outperform it. New untouched data or nested candidate selection is required for a selected-champion claim.'),
    ])
    notebook('03_Backtesting_and_Evaluation.ipynb', [
        ('md', '# OOF evidence, uncertainty and horizon scenarios\n\nThis is cross-sectional grouped validation, not historical back-testing. Contract expiry year and expansion tags are proxy segments rather than product families.'),
        ('code', SETUP),
        ('code', """metrics = pd.read_csv(run_dir / 'metrics.csv').set_index('model')
ci = pd.read_csv(run_dir / 'bootstrap_intervals.csv')
display(metrics.loc[['static_5', 'training_prevalence', PRIMARY_SCORE]])
display(ci[ci['model'] == PRIMARY_SCORE])
from src.backtesting import improvement_text
print(improvement_text(metrics.loc[PRIMARY_SCORE, 'brier_lift_vs_5_pct']))
print('A gain over 5% may only correct the portfolio rate; compare with training prevalence.')
rate_comparison = ci.loc[(ci.model == PRIMARY_SCORE) & (ci.baseline == 'training_prevalence')].iloc[0]
if rate_comparison.gain_ci_low > 0:
    print('The fixed-prediction interval supports lower Brier than training prevalence.')
else:
    print('No established Brier advantage over training prevalence: the interval includes zero or worse.')
print('Candidate ROC-AUC:', round(float(metrics.loc[PRIMARY_SCORE, 'roc_auc']), 6))
"""),
        ('md', '## Paired account uncertainty\n\n1,000 resamples preserve sibling arrays. These intervals condition on saved predictions and omit refitting/model-selection uncertainty. Positive gain means lower Brier.'),
        ('code', """rows = ci[(ci['baseline'] == 'training_prevalence') & (ci['model'] != 'training_prevalence')].copy()
fig, axes = plt.subplots(1, 2, figsize=(14, max(4, .4 * len(rows))))
close_view = rows.loc[rows.model.str.endswith('sigmoid') | rows.model.eq('logistic')]
for ax, subset, title in zip(axes, [rows, close_view],
                              ['All candidates', 'Mapped candidates and logistic: expanded scale']):
    ax.errorbar(subset.brier_gain, np.arange(len(subset)),
                xerr=[subset.brier_gain - subset.gain_ci_low, subset.gain_ci_high - subset.brier_gain],
                fmt='o', color='#245b8a', capsize=4)
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_yticks(np.arange(len(subset)), subset['model'], fontsize=8)
    ax.set(xlabel='Brier gain versus learned portfolio rate', title=title)
fig.suptitle('Paired account bootstrap 95% intervals')
plt.tight_layout()
plt.show()
"""),
        ('code', """cohorts = pd.read_csv(run_dir / 'cohorts.csv')
buckets = pd.read_csv(run_dir / 'historical_exposure_buckets.csv')
display(cohorts)
display(buckets)
assert np.isclose(buckets.signed_historical_profit_usd.sum(), oof.historical_15yr_net_profit_usd.sum())
print('Dollar quantities above are signed historical exposure, not future loss or discounted CLV.')
"""),
        ('md', 'Reporting preserves the sign of Brier improvement: a negative lift is deterioration. Empty exposure buckets have no observed event rate. Cohorts are descriptive diagnostics; neither contract year nor expansion category supplies product-family or matured observation-window evidence. OOF evidence is keyed, full precision and hash checked; full-fit production scores never replace it.'),
        ('md', '## Assumed years 2-15\n\nOnly the binary 12-month label is observed. Constant annual churn gives S(t)=(1-p)^t. The following plot and unit-cash-flow sensitivity are explicit scenarios, not a learned survival curve or dollar impact. Event/censoring history and time-indexed future cash flows are unavailable.'),
        ('code', """from src.scenarios import survival_scenario, discounted_clv_scenario
years = np.arange(1, 16)
rates = {'static 5%': .05, 'portfolio prevalence': float(oof[TARGET].mean()),
         'candidate mean forecast': float(oof[PRIMARY_SCORE].mean())}
for name, rate in rates.items():
    plt.plot(years, survival_scenario(rate), label=name)
plt.xlabel('Assumed year')
plt.ylabel('Assumed survival probability')
plt.title('Constant-churn horizon sensitivity')
plt.legend()
plt.tight_layout()
plt.show()
display(pd.DataFrame([
    {'scenario': name, 'discount_rate': r,
     'value_per_one_unit_annual_cashflow': float(discounted_clv_scenario(np.ones(15), p, r))}
    for name, p in rates.items() for r in [0, .05, .10]
]))
assert np.isclose(survival_scenario(.05)[-1], .463291230159753)
assert np.all(survival_scenario(0) == 1) and np.all(survival_scenario(1) == 0)
assert discounted_clv_scenario([-100, -100], 0, 0) == -200
"""),
        ('md', '## Final scoring delivery and integrity\n\nScoring loads the persisted transformer, unweighted models and probability maps. XGB/RF columns are individually sigmoid mapped. The ensemble maps 0.6 × raw XGB + 0.4 × raw RF; it is not a weighted mixture of already mapped columns. The immutable score directory contains the exact 25-column export, validated input snapshot, manifest and complete component explanations. Pre-export checks include warehouse precision, anomaly flags, probabilities, keys, dates and bucket consistency after rounding.'),
        ('code', """from src.config import SCORE_COLUMNS
from src.contracts import validate_scored_output
pointer = json.loads((OUTPUTS / 'latest_score_run.json').read_text())
score_dir = OUTPUTS / pointer['directory']
score_manifest = json.loads((score_dir / 'score_manifest.json').read_text())
assert score_manifest['training_run_id'] == manifest['run_id']
assert score_manifest['bundle_sha256'] == manifest['bundle_sha256']
for filename, field in [('CLV_SCORED_OUTPUT.csv', 'csv_sha256'),
                        ('input_snapshot.csv', 'snapshot_sha256'),
                        ('xgb_explanations.json', 'explanations_sha256')]:
    assert sha256_file(score_dir / filename) == score_manifest[field]
scored = pd.read_csv(score_dir / 'CLV_SCORED_OUTPUT.csv', dtype={'account_id': str, 'array_id': str})
validate_scored_output(scored)
snapshot = validate_raw_frame(score_dir / 'input_snapshot.csv')
reproduced = predict_bundle(bundle, snapshot)
for component in ['xgb', 'rf', 'ensemble']:
    np.testing.assert_allclose(scored[f'{component}_churn_probability'], reproduced[component], atol=5.1e-7)
assert sha256_file(OUTPUTS / 'CLV_SCORED_OUTPUT.csv') == score_manifest['csv_sha256']
assert sha256_file(OUTPUTS / 'backtesting_report.md') == sha256_file(run_dir / 'backtesting_report.md')
display(pd.Series({k: score_manifest[k] for k in [
    'score_run_id', 'training_run_id', 'model_version', 'rows', 'scored_at_utc',
    'reference_date', 'target_horizon', 'shap_units', 'deployment_status'
]}).to_frame('value'))
display(pd.DataFrame({'position': np.arange(1, len(SCORE_COLUMNS) + 1), 'export_column': SCORE_COLUMNS}))
display(scored.head(3))
print('PASS: scoring lineage, hashes, exact export contract and reproduced probabilities.')
"""),
        ('md', '## Account array-event summaries\n\nAccount aggregation reports conditional probability bounds for any/all array events without an independence assumption. These are not account-cancellation probabilities. Incomplete array prediction coverage withholds bounds; incomplete label coverage withholds observed proxies. The OOF summary includes coverage for the full supplied population, including accounts with no labeled arrays.'),
        ('code', """analytics_dir = score_dir / 'submission_analytics'
analytics_manifest = json.loads((analytics_dir / 'analytics_manifest.json').read_text())
assert analytics_manifest['training_run_id'] == manifest['run_id']
assert analytics_manifest['score_manifest_sha256'] == sha256_file(score_dir / 'score_manifest.json')
for filename, metadata in analytics_manifest['artifacts'].items():
    assert sha256_file(analytics_dir / filename) == metadata['sha256']
accounts = pd.read_csv(analytics_dir / 'account_array_event_summary.csv')
account_oof = pd.read_csv(analytics_dir / 'account_oof_array_event_summary.csv')
display(accounts.head(5))
display(account_oof[['complete_probability_coverage', 'complete_label_coverage']].value_counts().to_frame('accounts'))
assert len(accounts) == snapshot.account_id.nunique()
assert np.isclose(accounts.signed_historical_profit_usd.sum(), snapshot.historical_15yr_net_profit_usd.sum())
print(analytics_manifest['account_semantics'])
"""),
        ('md', '## Capacity diagnostics and per-array 15-year delivery\n\nCapacity headroom/pressure values are arithmetic at the reported current load, preserving supplied tags and anomaly flags. Out-of-range utilization may be a data defect; no future expansion is inferred from it. Each array has a constant-rate 15-year scenario compared with 5%. Dollar scenarios require explicit future cash flows; the delivered default uses one unit annually. The learned binary array event does not establish absorbing loss or commercial renewal.'),
        ('code', """capacity = pd.read_csv(analytics_dir / 'capacity_expansion_signals.csv')
horizons = pd.read_csv(analytics_dir / 'fifteen_year_scenarios.csv')
cash_values = pd.read_csv(analytics_dir / 'discounted_cash_flow_scenarios.csv')
display(capacity.head(5))
display(horizons[['account_id', 'array_id', 'scenario', 'assumed_annual_event_probability',
                  'assumed_survival_year_1', 'assumed_survival_year_5',
                  'assumed_survival_year_15', 'assumed_event_by_year_15']].head(5))
display(cash_values.groupby(['scenario', 'discount_rate', 'value_units']).scenario_discounted_value.agg(['count', 'mean']))
assert len(horizons) == 2 * len(snapshot)
for year in range(1, 16):
    np.testing.assert_allclose(horizons[f'assumed_survival_year_{year}'],
                               (1 - horizons.assumed_annual_event_probability)**year, atol=1e-14)
print(analytics_manifest['horizon_semantics'])
print(analytics_manifest['cash_flow_semantics'])
"""),
        ('md', '## Guarded future-data workflows\n\nThe dated workflow supports array-event, actual account-cancellation and realized capacity-expansion targets once their business definitions and snapshot/outcome dates are supplied. Training labels must be available before each cutoff; validation outcomes must have matured by evaluation-as-of. Each window fits its own preprocessing and grouped calibration. Supplied expansion tags are excluded when predicting future expansion. Real product-family mappings enable additional XGB/RF/ensemble cohort comparisons; expiry years and expansion tags are not renamed product families. Header-only contracts are documented in docs/REMAINING_DATA_CONTRACTS.md. Synthetic regression fixtures demonstrate software behavior only.'),
        ('code', """from src.temporal_validation import validate_history
try:
    validate_history(DATA_PATH)
except ValueError as error:
    print('Expected data-availability guard:', error)
else:
    raise AssertionError('Cross-sectional input must not pass as historical evidence')
print('Real product-family evaluation:', analytics_manifest['product_cohort_status'])
display(pd.DataFrame([
    {'contract': path.name, 'fields': ', '.join(pd.read_csv(path).columns), 'observations': len(pd.read_csv(path))}
    for path in sorted((ROOT / 'data/contracts').glob('*.csv'))
]))
print('No real historical/cancellation/expansion-outcome performance is claimed from these empty contracts.')
"""),
        ('md', '## Snowflake ingestion implementation and acceptance\n\nRun setup, DDL and the procedure script in a fresh selected sandbox. The fixed internal CLV_STAGE and CSV format match the 25 export columns. The six arguments are FILE_NAME, TARGET_MODEL_VERSION, RUN_ID, TRAINING_RUN_ID, SNAPSHOT_SHA256 and BUNDLE_SHA256. Filenames are validated relative paths. Each call uses its own temporary staging table; one transaction writes immutable history, the physical CLV_SCORED_OUTPUT table and the run ledger. Older timestamps cannot replace newer current scores; ties use lexical RUN_ID order, consistent with the reconciliation view. Identical payload/lineage replays return REPLAY_UNCHANGED; changed same-run payloads are rejected. Failures roll back and are logged. Standard-table keys are informational: serialize ingestion through one writer; concurrent same-run idempotency is not established.'),
        ('code', """ddl = (ROOT / 'sql/ddl_clv_scored_output.sql').read_text(encoding='utf-8-sig')
body = ddl.split('CREATE TABLE IF NOT EXISTS CLV_SCORE_STAGING_TEMPLATE (')[1].split(');')[0]
assert [line.strip().split()[0].lower() for line in body.splitlines() if line.strip()] == SCORE_COLUMNS
assert 'CREATE TABLE IF NOT EXISTS CLV_SCORED_OUTPUT LIKE CLV_SCORED_HISTORY' in ddl
handler = (ROOT / 'sql/stored_proc_scoring_pipeline.sql').read_text(encoding='utf-8-sig').split('$$')[1]
compile(handler, 'snowflake_handler', 'exec')
assert 'MERGE INTO CLV_SCORED_OUTPUT' in handler
print('PASS: local SQL column order and embedded Python syntax only. Live execution remains unverified.')
print((ROOT / 'sql/sandbox_acceptance.sql').read_text(encoding='utf-8-sig'))
"""),
        ('md', '## Reproduction and final checks\n\nCPython 3.12.10 and requirements-lock.txt pin the tested environment. From the project root, run `python -m unittest discover -s tests -p "test_*.py" -v` and `python tests/verify_all_steps.py`. The regression suite covers input rejection, fold isolation, persisted state, hash/lineage tampering, unsupported fitted evidence, report preservation, export rounding, SHAP reconstruction, account bounds/coverage, signed future cash flows, chronological maturity and target guards, genuine product mappings, and warehouse transaction/replay control flow. The verifier is read-only and does not regenerate predictions or overwrite reports. The three notebooks read the current saved artifacts; the maintained generator is tools/build_notebooks.py. Archived submissions, earlier runs and legacy script aliases are excluded from the final delivery.'),
        ('md', '## Operational recommendation\n\nDo not claim behavioral targeting lift from probability calibration alone. Preserve immutable OOF and scoring lineage, retain the prevalence challenger, obtain mature longitudinal outcomes, and run Snowflake sandbox acceptance before deployment. SQL static checks are not evidence of warehouse execution.'),
    ])
    print('Built three evidence-reading notebooks.')


if __name__ == '__main__':
    main()

