"""Business-semantic, chronological and transaction-contract regression checks."""
import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.config import ROOT, DATA_PATH, PRIMARY_SCORE
from src.data_cleaning import TARGET
from src.submission_analytics import account_summary, capacity_signals, horizon_scenarios, product_cohort_metrics, generate_submission_analytics
from src.model_training import train_and_evaluate_models, sha256_file
from src.scoring_pipeline import generate_production_scores
from src.temporal_validation import validate_history, chronological_split, rolling_origin_backtest


def snapshot():
    return pd.DataFrame([
        {'account_id': f'A{i}', 'array_id': f'X{i}_{j}', 'contract_end_date': '2027-04-01',
         'historical_15yr_net_profit_usd': -100 if i == 0 else 1000,
         'current_capacity_tb': 100., 'utilization_rate_pct': 1.45 if i == 0 else .5,
         'expansion_potential_signal': 'UNKNOWN' if i == 0 else 'Low', TARGET: j}
        for i in range(12) for j in range(2)
    ])


def history():
    records = []
    for year in (2018, 2020, 2022):
        for i in range(12):
            records.append({
                'account_id': f'A{i}', 'array_id': f'X{i}', 'contract_end_date': f'{year+2}-06-01',
                'historical_15yr_net_profit_usd': 1000 * (i + 1), 'current_capacity_tb': float(i + 1),
                'utilization_rate_pct': .2 + i / 20, 'expansion_potential_signal': 'High' if i % 2 else 'Low',
                TARGET: i % 2, 'observation_date': f'{year}-01-01', 'feature_as_of': f'{year}-01-01',
                'label_window_end': f'{year+1}-01-01', 'label_available_at': f'{year+1}-01-02',
            })
    return pd.DataFrame(records)


class AccountAndScenarioTests(unittest.TestCase):
    def test_account_bounds_do_not_assume_independence(self):
        raw = snapshot().iloc[:2]
        probabilities = raw[['account_id', 'array_id']].assign(p=[.2, .7])
        summary = account_summary(raw, probabilities, 'p', 'test_fixture').iloc[0]
        self.assertAlmostEqual(summary.any_array_event_probability_lower_bound, .7)
        self.assertAlmostEqual(summary.any_array_event_probability_upper_bound, .9)
        self.assertAlmostEqual(summary.all_arrays_event_probability_lower_bound, 0)
        self.assertAlmostEqual(summary.all_arrays_event_probability_upper_bound, .2)
        self.assertEqual(summary.observed_any_array_event_proxy, 1)
        self.assertEqual(summary.observed_all_arrays_event_proxy, 0)
        self.assertEqual(summary.signed_historical_profit_usd, -200)
        self.assertNotIn('account_churn_probability', summary.index)

    def test_missing_arrays_or_labels_do_not_become_non_events(self):
        raw = snapshot().iloc[:2].copy()
        raw.loc[1, TARGET] = np.nan
        partial = raw.iloc[:1][['account_id', 'array_id']].assign(p=.2)
        result = account_summary(raw, partial, 'p', 'grouped_oof').iloc[0]
        self.assertFalse(result.complete_probability_coverage)
        self.assertTrue(pd.isna(result.any_array_event_probability_upper_bound))
        self.assertFalse(result.complete_label_coverage)
        self.assertTrue(pd.isna(result.observed_any_array_event_proxy))

    def test_account_rejects_unknown_keys_duplicates_and_bad_probabilities(self):
        raw = snapshot().iloc[:2]
        probabilities = raw[['account_id', 'array_id']].assign(p=.2)
        for bad in [pd.concat([probabilities, probabilities]),
                    probabilities.assign(p=np.nan), probabilities.assign(p=1.1),
                    probabilities.assign(array_id='unknown')]:
            with self.assertRaises(ValueError):
                account_summary(raw, bad, 'p', 'fixture')

    def test_capacity_signals_are_current_load_arithmetic(self):
        signals = capacity_signals(snapshot())
        self.assertAlmostEqual(signals.iloc[0].additional_capacity_to_reach_100pct_tb, 45)
        self.assertAlmostEqual(signals.iloc[2].spare_capacity_at_reported_load_tb, 50)
        self.assertTrue(signals.iloc[0].expansion_signal_unknown)

    def test_unit_scenarios_and_probability_boundaries(self):
        curves, values = horizon_scenarios(snapshot().iloc[:2], [0, 1], discount_rates=[0])
        fitted = curves.loc[curves.scenario.eq('calibrated_array_event_rate')]
        self.assertEqual(fitted.iloc[0].assumed_survival_year_15, 1)
        self.assertEqual(fitted.iloc[1].assumed_survival_year_15, 0)
        fitted_values = values.loc[values.scenario.eq('calibrated_array_event_rate')]
        self.assertEqual(fitted_values.scenario_discounted_value.tolist(), [15, 0])
        self.assertTrue(values.value_units.eq('one_unit_annual_cash_flow').all())
        for probabilities in [[.1], [np.nan, .1], [1.1, .1]]:
            with self.assertRaises(ValueError):
                horizon_scenarios(snapshot().iloc[:2], probabilities)

    def test_explicit_future_cash_flows_preserve_negative_values(self):
        raw = snapshot().iloc[:1]
        flows = pd.DataFrame({'account_id': ['A0'] * 15, 'array_id': ['X0_0'] * 15,
                              'year': range(1, 16), 'annual_cash_flow_usd': [-10] * 15})
        _, values = horizon_scenarios(raw, [0], cash_flows=flows, discount_rates=[0])
        self.assertEqual(values.iloc[0].scenario_discounted_value, -150)
        self.assertTrue(values.value_units.eq('usd_from_explicit_user_cash_flows').all())
        for bad in [flows.iloc[:-1], pd.concat([flows, flows.iloc[:1]]), flows.assign(array_id='other')]:
            with self.assertRaises(ValueError):
                horizon_scenarios(raw, [0], cash_flows=bad)

    def test_real_product_mapping_is_required_and_metrics_cover_both_baselines(self):
        raw = snapshot()
        evidence = raw.assign(static_5=.05, training_prevalence=.5,
                              unweighted_xgb_sigmoid=.4, unweighted_rf_sigmoid=.6, prediction_source='grouped_oof')
        evidence[PRIMARY_SCORE] = .5
        mapping = raw[['account_id', 'array_id']].assign(product_family='P1')
        metrics = product_cohort_metrics(evidence, mapping)
        self.assertEqual(len(metrics), 6)
        self.assertEqual(set(metrics.baseline), {'static_5', 'training_prevalence'})
        for bad in [mapping.iloc[:-1], pd.concat([mapping, mapping.iloc[:1]]), mapping.assign(product_family='')]:
            with self.assertRaises(ValueError):
                product_cohort_metrics(evidence, bad)
        with self.assertRaisesRegex(ValueError, 'grouped OOF'):
            product_cohort_metrics(evidence.assign(prediction_source='full_fit_scoring'), mapping)


class ChronologicalValidationTests(unittest.TestCase):
    def test_cross_sectional_case_study_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Dated validation needs'):
            validate_history(DATA_PATH)

    def test_feature_availability_and_label_maturity_are_enforced(self):
        for column, value in [('feature_as_of', '2019-01-01'),
                              ('label_window_end', '2018-12-01'),
                              ('label_available_at', '2018-01-02')]:
            frame = history()
            frame.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_history(frame)
        frame = history()
        frame.loc[0, 'label_available_at'] = None
        with self.assertRaises(ValueError):
            validate_history(frame)

    def test_duplicate_snapshots_rejected(self):
        frame = history()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_history(pd.concat([frame, frame.iloc[:1]]))

    def test_cutoff_excludes_unavailable_labels_and_uses_latest_entity_snapshot(self):
        frame = validate_history(history())
        train, val = chronological_split(frame, '2022-01-01', '2024-01-01')
        self.assertEqual(len(train), 12)
        self.assertTrue(train.observation_date.eq(pd.Timestamp('2020-01-01')).all())
        self.assertTrue((train.label_available_at < pd.Timestamp('2022-01-01')).all())
        self.assertTrue(val.observation_date.eq(pd.Timestamp('2022-01-01')).all())
        with self.assertRaises(ValueError):
            chronological_split(frame, '2022-01-01', '2022-06-01')

    def test_expansion_tag_is_excluded_for_future_expansion_target(self):
        frame = history().rename(columns={TARGET: 'capacity_expanded_next_12m'})
        validated = validate_history(frame, 'capacity_expanded_next_12m')
        self.assertTrue(validated.expansion_potential_signal.eq('UNKNOWN').all())
        without_tag = validate_history(frame.drop(columns='expansion_potential_signal'), 'capacity_expanded_next_12m')
        self.assertTrue(without_tag.expansion_potential_signal.eq('UNKNOWN').all())

    def test_expansion_temporal_fixture_uses_prevalence_not_static_churn_baseline(self):
        frame = history().rename(columns={TARGET: 'capacity_expanded_next_12m'})
        with tempfile.TemporaryDirectory() as temp:
            _, predictions, metrics = rolling_origin_backtest(
                frame, ['2020-01-01'], '2024-01-01', 'Synthetic expansion fixture',
                target='capacity_expanded_next_12m', output_root=temp,
                params={'xgb_estimators': 3, 'rf_estimators': 5})
            self.assertNotIn('static_5', predictions)
            self.assertIn('capacity_expanded_next_12m', predictions)
            self.assertEqual(len(metrics), 4)

    def test_account_temporal_fixture_has_one_row_per_account_and_dated_cohorts(self):
        frame = history().drop(columns='array_id').rename(columns={TARGET: 'account_cancelled_next_12m'})
        frame['product_family'] = 'test_product'
        with tempfile.TemporaryDirectory() as temp:
            root, predictions, _ = rolling_origin_backtest(
                frame, ['2020-01-01'], '2024-01-01', 'Synthetic account cancellation fixture',
                target='account_cancelled_next_12m', entity_level='account', output_root=temp,
                params={'xgb_estimators': 3, 'rf_estimators': 5})
            self.assertNotIn('array_id', predictions)
            self.assertIn('account_cancelled_next_12m', predictions)
            self.assertEqual(len(predictions), 12)
            self.assertTrue((root / 'historical_product_cohorts.csv').exists())

    def test_account_snapshots_require_account_outcome_and_unique_account_date(self):
        frame = history().drop(columns='array_id').rename(columns={TARGET: 'account_cancelled_next_12m'})
        validated = validate_history(frame, 'account_cancelled_next_12m', 'account')
        self.assertTrue(validated.array_id.eq(validated.account_id).all())
        with self.assertRaises(ValueError):
            validate_history(frame, 'account_cancelled_next_12m', 'array')

    def test_rolling_origin_fixture_outputs_are_chronological_not_main_oof(self):
        with tempfile.TemporaryDirectory() as temp:
            root, predictions, metrics = rolling_origin_backtest(
                history(), ['2020-01-01', '2022-01-01'], '2024-01-01',
                'Synthetic test outcome only; not case-study evidence', output_root=temp,
                inner_folds=3, params={'xgb_estimators': 3, 'rf_estimators': 5})
            self.assertEqual(len(predictions), 24)
            self.assertTrue(predictions.prediction_source.eq('chronological_holdout').all())
            self.assertEqual(len(metrics), 10)
            manifest = json.loads((root / 'manifest.json').read_text())
            for window in manifest['windows']:
                self.assertLess(pd.Timestamp(window['latest_training_label_available_at']), pd.Timestamp(window['cutoff']))
            self.assertFalse((Path(temp) / 'latest_run.json').exists())


class SubmissionArtifactTests(unittest.TestCase):
    def test_score_pointer_round_trip_on_windows_and_posix(self):
        from src.pipeline import score
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp)
            path = outputs / 'score_runs' / ('a' * 32) / 'CLV_SCORED_OUTPUT.csv'
            path.parent.mkdir(parents=True)
            path.write_text('fixture only\n', encoding='utf-8')
            with patch('src.pipeline.OUTPUTS', outputs), \
                 patch('src.pipeline.generate_production_scores', return_value=(pd.DataFrame(), {'score_run_id': 'a' * 32}, path)), \
                 patch('src.pipeline.generate_submission_analytics'):
                score()
            pointer = json.loads((outputs / 'latest_score_run.json').read_text())
            parts = PurePosixPath(pointer['directory']).parts
            self.assertEqual(parts, ('score_runs', 'a' * 32))
            self.assertEqual((outputs.joinpath(*parts) / path.name).read_bytes(), path.read_bytes())

    def test_derived_delivery_is_keyed_hashed_and_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = snapshot()
            trained = train_and_evaluate_models(raw, n_splits=3, inner_splits=2, output_root=temp,
                                                params={'xgb_estimators': 3, 'rf_estimators': 5})
            _, _, score_path = generate_production_scores(raw, trained['run_dir'],
                                                          Path(temp) / 'score' / 'CLV_SCORED_OUTPUT.csv')
            mapping = raw[['account_id', 'array_id']].assign(product_family='fixture_only')
            root, manifest = generate_submission_analytics(score_path.parent, trained['run_dir'], product_mapping=mapping)
            self.assertEqual(len(pd.read_csv(root / 'account_array_event_summary.csv')), 12)
            self.assertEqual(len(pd.read_csv(root / 'fifteen_year_scenarios.csv')), 48)
            self.assertEqual(len(pd.read_csv(root / 'discounted_cash_flow_scenarios.csv')), 144)
            self.assertTrue((root / 'product_family_cohorts.csv').exists())
            self.assertEqual(manifest['training_run_id'], trained['manifest']['run_id'])
            for file, info in manifest['artifacts'].items():
                self.assertEqual(sha256_file(root / file), info['sha256'])
            with self.assertRaises(FileExistsError):
                generate_submission_analytics(score_path.parent, trained['run_dir'])
            with score_path.open('a') as output:
                output.write('\nchanged')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                generate_submission_analytics(score_path.parent, trained['run_dir'])


class ProcedureTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sql = (ROOT / 'sql/stored_proc_scoring_pipeline.sql').read_text(encoding='utf-8-sig')
        cls.namespace = {}
        exec(compile(sql.split('$$')[1], 'snowflake_handler', 'exec'), cls.namespace)

    def session(self, replay=False, changed=False, fail_latest=False):
        class FakeSession:
            def __init__(self):
                self.calls = []

            def sql(self, text, params=None):
                owner = self
                class Query:
                    def collect(self):
                        owner.calls.append((text, params))
                        if text.startswith('SELECT LAST_QUERY_ID'):
                            return [['fake_query_id']]
                        if text.startswith('SELECT TRAINING_RUN_ID'):
                            return [('1'*32, 'v2', 'fixture.csv', 'a'*64, 'b'*64, 2)] if replay else []
                        if text.startswith('SELECT COUNT(*) FROM (('):
                            return [[int(changed)]]
                        if text.startswith('SELECT COUNT(*) FROM CLV_SCORED_HISTORY'):
                            return [[2]]
                        if text.startswith('SELECT COUNT(*) FROM (SELECT') or 'WHERE MODEL_VERSION' in text:
                            return [[0]]
                        if text.startswith('SELECT COUNT(*) FROM TMP_CLV_'):
                            return [[2]]
                        if text.startswith('MERGE INTO CLV_SCORED_HISTORY'):
                            return [[2]]
                        if text.startswith('MERGE INTO CLV_SCORED_OUTPUT') and fail_latest:
                            raise RuntimeError('Injected current-score merge failure')
                        return []
                return Query()
        return FakeSession()

    def invoke(self, session):
        return self.namespace['ingest'](session, 'fixture.csv', 'v2', '0'*32, '1'*32, 'a'*64, 'b'*64)

    def test_history_current_scores_and_ledger_commit_together(self):
        session = self.session()
        result = self.invoke(session)
        queries = [text for text, _ in session.calls]
        position = lambda prefix: next(i for i, text in enumerate(queries) if text.startswith(prefix))
        self.assertLess(position('BEGIN TRANSACTION'), position('MERGE INTO CLV_SCORED_HISTORY'))
        self.assertLess(position('MERGE INTO CLV_SCORED_HISTORY'), position('MERGE INTO CLV_SCORED_OUTPUT'))
        self.assertLess(position('MERGE INTO CLV_SCORED_OUTPUT'), position('INSERT INTO CLV_SCORE_RUNS'))
        self.assertLess(position('INSERT INTO CLV_SCORE_RUNS'), position('COMMIT'))
        latest = queries[position('MERGE INTO CLV_SCORED_OUTPUT')]
        self.assertIn('s.SCORED_AT > t.SCORED_AT', latest)
        self.assertIn('s.RUN_ID > t.RUN_ID', latest)
        self.assertIn('latest_merge_query_id', result)

    def test_failed_current_merge_rolls_back_history_and_never_commits(self):
        session = self.session(fail_latest=True)
        with self.assertRaisesRegex(RuntimeError, 'Injected'):
            self.invoke(session)
        queries = [text for text, _ in session.calls]
        self.assertIn('ROLLBACK', queries)
        self.assertNotIn('COMMIT', queries)
        self.assertFalse(any(text.startswith('INSERT INTO CLV_SCORE_RUNS') for text in queries))
        self.assertTrue(any(text.startswith('INSERT INTO CLV_SCORE_FAILURES') for text in queries))
        self.assertTrue(queries[-1].startswith('DROP TABLE IF EXISTS TMP_CLV_'))

    def test_unchanged_replay_has_no_history_or_current_writes(self):
        session = self.session(replay=True)
        self.assertEqual(self.invoke(session)['status'], 'REPLAY_UNCHANGED')
        self.assertFalse(any(text.startswith('MERGE') for text, _ in session.calls))

    def test_changed_replay_is_rejected_before_transaction(self):
        session = self.session(replay=True, changed=True)
        with self.assertRaisesRegex(ValueError, 'changed scores'):
            self.invoke(session)
        self.assertFalse(any(text == 'BEGIN TRANSACTION' for text, _ in session.calls))


if __name__ == '__main__':
    unittest.main()
