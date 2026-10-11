"""Regression tests for the review's real failure modes (standard-library unittest)."""
import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import ROOT, PRIMARY_SCORE, SCORE_COLUMNS
from src.data_cleaning import validate_raw_frame, TARGET
from src.feature_engineering import CLVPreprocessor
from src.model_training import (train_and_evaluate_models, grouped_splits, load_bundle, predict_bundle)
from src.backtesting import (validate_evaluation, read_evidence, run_backtesting_suite,
                             paired_account_ci, improvement_text)
from src.scoring_pipeline import generate_production_scores
from src.scenarios import survival_scenario, discounted_clv_scenario
from src.explainability import xgb_contributions
from src.utils import assign_risk_bucket
from src.contracts import validate_scored_output


def fixture():
    return pd.DataFrame([
        {'account_id': f'A{i}', 'array_id': f'X{i}_{j}', 'contract_end_date': '2027-04-01',
         'historical_15yr_net_profit_usd': (-100 if i == 0 else 1000 * (i + 1)),
         'current_capacity_tb': float(i + j + 1), 'utilization_rate_pct': .5 + .05 * i,
         'expansion_potential_signal': ['UNKNOWN', 'Low', 'Medium', 'High'][i % 4],
         TARGET: j}
        for i in range(12) for j in range(2)
    ])


class InputAndFeatureTests(unittest.TestCase):
    def test_invalid_capacity_labels_utilization_and_dates(self):
        for column, value in [('current_capacity_tb', 0), (TARGET, 2), (TARGET, .5),
                              ('utilization_rate_pct', -.1), ('current_capacity_tb', np.inf),
                              ('contract_end_date', None), ('expansion_potential_signal', 'Typo')]:
            with self.subTest(column=column, value=value):
                df = fixture()
                if column == TARGET:
                    df[TARGET] = df[TARGET].astype(float)
                df.loc[0, column] = value
                with self.assertRaises((ValueError, TypeError)):
                    validate_raw_frame(df)

    def test_duplicate_and_missing_keys(self):
        with self.assertRaises(ValueError):
            validate_raw_frame(pd.concat([fixture(), fixture().iloc[:1]]))
        df = fixture()
        df.loc[0, 'account_id'] = None
        with self.assertRaises(ValueError):
            validate_raw_frame(df)

    def test_reject_precomputed_feature_api(self):
        with self.assertRaisesRegex(ValueError, 'raw DataFrame'):
            validate_raw_frame(np.ones((10, 12)))

    def test_csv_preserves_leading_zero_identifiers(self):
        df = fixture()
        df['account_id'] = ['000' + str(i) for i in range(len(df))]
        df['array_id'] = ['00' + str(i) for i in range(len(df))]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'raw.csv'
            df.to_csv(path, index=False)
            loaded = validate_raw_frame(path)
            self.assertEqual(loaded.account_id.tolist(), df.account_id.tolist())
            self.assertEqual(loaded.array_id.tolist(), df.array_id.tolist())

    def test_nominal_unknown_and_feature_allowlist(self):
        df = fixture()
        df['accidental_target_copy'] = df[TARGET]
        prep = CLVPreprocessor().fit(df)
        X = prep.transform(df)
        self.assertEqual(X.shape[1], 15)
        self.assertNotIn('accidental_target_copy', prep.feature_names_)
        self.assertNotIn(TARGET, prep.feature_names_)
        idx = prep.feature_names_.index('expansion_unknown')
        self.assertEqual(X[0, idx], 1)
        for name in ['expansion_low', 'expansion_medium', 'expansion_high']:
            self.assertEqual(X[0, prep.feature_names_.index(name)], 0)

    def test_fold_local_bins_and_no_refit_on_new_snapshot(self):
        df = fixture()
        prep = CLVPreprocessor().fit(df.iloc[:12])
        expected = np.unique(np.quantile(df.iloc[:12].current_capacity_tb, [.25, .5, .75]))
        np.testing.assert_array_equal(prep.capacity_edges_[1:-1], expected)
        edges = prep.capacity_edges_.copy()
        new = df.iloc[12:].drop(columns=TARGET).copy()
        new['current_capacity_tb'] = 10000
        new['historical_15yr_net_profit_usd'] = -1e7
        X = prep.transform(new)
        np.testing.assert_array_equal(prep.capacity_edges_, edges)
        self.assertTrue(np.isfinite(X).all())
        self.assertTrue((X[:, prep.feature_names_.index('signed_log_profit')] < 0).all())

    def test_constant_capacity_and_serialization(self):
        df = fixture().assign(current_capacity_tb=5)
        prep = CLVPreprocessor().fit(df)
        clone = pickle.loads(pickle.dumps(prep))
        np.testing.assert_array_equal(prep.transform(df), clone.transform(df))


class NestedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.df = fixture()
        cls.trained = train_and_evaluate_models(
            cls.df, n_splits=3, inner_splits=2, output_root=Path(cls.temp.name),
            params={'xgb_estimators': 3, 'rf_estimators': 5},
        )
        cls.run_dir = cls.trained['run_dir']
        cls.evidence, cls.manifest = read_evidence(cls.run_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_account_isolation_and_one_oof_prediction(self):
        validate_evaluation(self.evidence, self.manifest)
        self.assertEqual(len(self.evidence), len(self.df))
        for tr, va in grouped_splits(validate_raw_frame(self.df), 3):
            self.assertFalse(set(self.df.iloc[tr].account_id) & set(self.df.iloc[va].account_id))
        self.assertEqual(self.evidence.groupby('account_id').fold_id.nunique().max(), 1)

    def test_validation_manifest_rejects_overlap(self):
        bad = copy.deepcopy(self.manifest)
        bad['folds'][0]['train_accounts'].append(bad['folds'][0]['validation_accounts'][0])
        with self.assertRaisesRegex(ValueError, 'isolation'):
            validate_evaluation(self.evidence, bad)

    def test_rejects_fitted_production_prediction_source(self):
        bad = self.evidence.copy()
        bad['prediction_source'] = 'fitted_production'
        with self.assertRaisesRegex(ValueError, 'Fitted-production'):
            validate_evaluation(bad, self.manifest)

    def test_prevalence_baseline_rejects_validation_label_use(self):
        bad = copy.deepcopy(self.manifest)
        bad['folds'][0]['training_prevalence'] = .99
        with self.assertRaisesRegex(ValueError, 'training labels only'):
            validate_evaluation(self.evidence, bad)

    def test_rejects_nan_out_of_range_and_duplicate_predictions(self):
        for value in [np.nan, -.1, 1.1]:
            bad = self.evidence.copy()
            bad.loc[0, PRIMARY_SCORE] = value
            with self.assertRaises(ValueError):
                validate_evaluation(bad, self.manifest)
        bad = self.evidence.copy()
        bad.loc[1, ['account_id', 'array_id']] = bad.loc[0, ['account_id', 'array_id']]
        with self.assertRaises(ValueError):
            validate_evaluation(bad, self.manifest)

    def test_hash_check_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')
            self.evidence.to_csv(root / 'evaluation_predictions.csv', index=False)
            with (root / 'evaluation_predictions.csv').open('a') as f:
                f.write('\nmodified')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                read_evidence(root)

    def test_bundle_reload_unlabeled_snapshot_and_probability_reconstruction(self):
        bundle, _ = load_bundle(self.run_dir)
        raw = self.df.drop(columns=TARGET)
        expected = predict_bundle(self.trained['bundle'], raw)
        actual = predict_bundle(bundle, raw)
        for name in expected:
            # Threaded tree reductions may differ at machine precision after reload.
            np.testing.assert_allclose(expected[name], actual[name], rtol=0, atol=1e-14)
        edges = bundle['preprocessor'].capacity_edges_.copy()
        out = Path(self.temp.name) / 'scoring' / 'CLV_SCORED_OUTPUT.csv'
        scored, metadata, _ = generate_production_scores(raw, run_dir=self.run_dir, output_path=out)
        self.assertEqual(scored.columns.tolist(), SCORE_COLUMNS)
        np.testing.assert_allclose(scored.ensemble_churn_probability, actual['ensemble'], atol=5e-7)
        np.testing.assert_array_equal(bundle['preprocessor'].capacity_edges_, edges)
        self.assertEqual(metadata['training_run_id'], self.manifest['run_id'])
        self.assertFalse(scored.isna().any().any())
        values, base, _ = xgb_contributions(bundle, raw)
        self.assertEqual(values.shape[1], 15)
        self.assertEqual(len(base), len(raw))
        with self.assertRaises(FileExistsError):
            generate_production_scores(raw, run_dir=self.run_dir, output_path=out)

    def test_report_deterioration_and_exposure_reconciliation(self):
        bad = self.evidence.copy()
        bad[PRIMARY_SCORE] = 1 - bad[TARGET]  # Deliberately terrible forecasts.
        with tempfile.TemporaryDirectory() as temp:
            result = run_backtesting_suite(bad, self.manifest, temp, repetitions=50)
            self.assertIn('Brier score deteriorates', result['report'])
            self.assertNotIn('Deploy ensemble', result['report'])
            self.assertAlmostEqual(result['buckets'].signed_historical_profit_usd.sum(),
                                   self.df.historical_15yr_net_profit_usd.sum())
        self.assertIn('deteriorates', improvement_text(-10.78))
        self.assertIn('improves', improvement_text(5))
        self.assertIn('unchanged', improvement_text(0))

    def test_scoring_rejects_export_precision_loss(self):
        raw = self.df.drop(columns=TARGET).copy()
        raw['current_capacity_tb'] = .001
        with self.assertRaisesRegex(ValueError, 'after rounding'):
            generate_production_scores(raw, run_dir=self.run_dir,
                                       output_path=Path(self.temp.name) / 'bad_precision.csv')
        self.assertFalse((Path(self.temp.name) / 'bad_precision.csv').exists())

    def test_paired_bootstrap_identical_baseline(self):
        ci = paired_account_ci(self.evidence, 'training_prevalence', 'training_prevalence', repetitions=50)
        self.assertEqual(ci['brier_gain'], 0)
        self.assertEqual(ci['gain_ci_low'], 0)
        self.assertEqual(ci['gain_ci_high'], 0)

    def test_reporting_cannot_overwrite_evidence_with_changed_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_backtesting_suite(self.evidence, self.manifest, temp, repetitions=50)
            before = (Path(temp) / 'backtesting_report.md').read_bytes()
            run_backtesting_suite(self.evidence, self.manifest, temp, repetitions=50)
            self.assertEqual(before, (Path(temp) / 'backtesting_report.md').read_bytes())
            modified = self.evidence.copy()
            modified[PRIMARY_SCORE] = 1 - modified[TARGET]
            with self.assertRaisesRegex(ValueError, 'immutable'):
                run_backtesting_suite(modified, self.manifest, temp, repetitions=50)
            self.assertEqual(before, (Path(temp) / 'backtesting_report.md').read_bytes())


class PolicyAndWarehouseTests(unittest.TestCase):
    def test_bucket_boundaries_and_invalid_probability(self):
        for p, bucket in [(0, 'LOW'), (.149999, 'LOW'), (.15, 'MEDIUM'),
                          (.399999, 'MEDIUM'), (.4, 'HIGH'), (.699999, 'HIGH'),
                          (.7, 'CRITICAL'), (1, 'CRITICAL')]:
            self.assertEqual(assign_risk_bucket(p), bucket)
        for p in [np.nan, np.inf, -.1, 1.1]:
            with self.assertRaises(ValueError):
                assign_risk_bucket(p)

    def test_scenario_boundaries_and_signed_cash_flows(self):
        np.testing.assert_array_equal(survival_scenario(0), np.ones(15))
        np.testing.assert_array_equal(survival_scenario(1), np.zeros(15))
        self.assertAlmostEqual(survival_scenario(.05)[-1], .463291230159753)
        self.assertEqual(discounted_clv_scenario([-10, -10], 0, 0), -20)
        for p in [np.nan, -.1, 1.1]:
            with self.assertRaises(ValueError):
                survival_scenario(p)

    def test_csv_ddl_column_order_and_handler_python_syntax(self):
        import re
        ddl = (ROOT / 'sql/ddl_clv_scored_output.sql').read_text(encoding='utf-8-sig')
        body = ddl.split('CREATE TABLE IF NOT EXISTS CLV_SCORE_STAGING_TEMPLATE (')[1].split(');')[0]
        cols = [line.strip().split()[0].lower() for line in body.splitlines() if line.strip()]
        self.assertEqual(cols, SCORE_COLUMNS)
        sql = (ROOT / 'sql/stored_proc_scoring_pipeline.sql').read_text(encoding='utf-8-sig')
        handler = sql.split('$$')[1]
        compile(handler, 'snowflake_handler', 'exec')  # Python syntax only, not warehouse compilation.
        namespace = {}
        exec(compile(handler, 'snowflake_handler', 'exec'), namespace)
        with self.assertRaises(ValueError):
            namespace['ingest'](None, "x.csv'); DROP TABLE t;--", 'v2', '0'*32, '1'*32, 'a'*64, 'b'*64)
        self.assertIn('FROM @CLV_STAGE', handler)
        self.assertNotIn('FROM @STAGE_PATH', handler)
        self.assertNotIn('TRUNCATE', handler)


if __name__ == '__main__':
    unittest.main()

