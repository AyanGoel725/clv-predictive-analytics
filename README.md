# Storage-array churn and CLV scenario case study

Public repository: [AyanGoel725/clv-predictive-analytics](https://github.com/AyanGoel725/clv-predictive-analytics).

This project estimates next-12-month **array-event** probability from a cross-sectional storage dataset. It provides account-grouped validation, calibrated XGBoost/Random Forest candidates, a portfolio-rate challenger, XGB explanations, account array-event summaries, capacity diagnostics, per-array 15-year scenarios, a guarded chronological evaluation workflow, and a Snowflake ingestion design.

**Status:** finalized local submission with verified evidence and scoring artifacts. Live Snowflake execution, account-cancellation semantics, label maturity, individual retention targeting and 15-year predictive accuracy are not established. The delivered run is identified below; `latest_run.json` and `latest_score_run.json` select its evidence and scoring artifacts. Archived submissions, superseded runs and redundant legacy scripts have been removed.

[Final readiness review](docs/SUBMISSION_READINESS.md) maps each mandate to the implemented evidence and remaining data/execution limits. The expanded suite passes 43 regression tests. The project is published and ready for PDF preparation with those limitations disclosed; final report assembly remains. Include the exact GitHub commit link in the PDF.

## Data and target contract

The supplied CSV contains 2,000 arrays belonging to 1,000 accounts. Training uses 1,884 labeled rows (298 events, 15.8174%); 116 missing labels are inference rows, never imputed labels or a test set. The event taxonomy is not supplied: one array retiring while an account renews must not be interpreted as losing that account. Both arrays leaving also does not establish commercial cancellation without a business rule. Observation timestamps and matured 12-month outcome windows must be supplied before historical back-testing.

Utilization is a fraction: 1.45 means 145%. There are 862 readings above 1, 58 negative-profit rows (minimum -$93,780), and 111 UNKNOWN expansion tags. Their causes are unknown. Utilization is capped with an anomaly flag; signed financial values are retained. Signed log handles profit without a learned minimum. UNKNOWN is nominal one-hot, not ordered below Low. Missing tags map to UNKNOWN; unrecognized category spellings are rejected. Capacity must be positive; invalid dates, labels, duplicate row keys, nonfinite inputs and negative utilization fail validation.

An expansion tag is an opportunity signal, not a supervised prediction of future expansion. Product-family analysis requires a lookup. Historical profit totals are $4,305,689,564 for all rows and $4,053,131,625 for labeled rows; they are not future cash flows, expected loss or measured CLV.

## Validation and modeling

Five outer GroupKFold folds keep every account intact. Each fold fits its own capacity quartiles; all 15 features have explicit order. Three inner grouped folds generate probabilities for sigmoid and isotonic calibration. Validation labels never fit a mapping. Both weighted and unweighted fixed XGB/RF candidates are evaluated; raw ensemble weights remain 0.6/0.4. Fold-trained prevalence and standardized logistic regression are challengers.

The prespecified scoring candidate is the **unweighted sigmoid ensemble**, not a winner selected on outer-fold results. No action threshold is tuned on outer labels. Compare Brier, log loss, AP (average precision), ROC-AUC, bias and both equal-width/equal-count ECE. Bootstrap intervals resample whole accounts, condition on fixed predictions and do not include model refitting or selection uncertainty. There is no untouched final test set; subsequent candidate selection requires nested selection or new data.

## Current results

<!-- RESULTS_START -->

Run: f0095e2660d447bd9df382f307062afb

| model                       |   mean_probability |      bias |    brier |   roc_auc |   average_precision |   brier_lift_vs_5_pct |
|:----------------------------|-------------------:|----------:|---------:|----------:|--------------------:|----------------------:|
| static_5                    |           0.050000 | -0.108174 | 0.144857 |  0.500000 |            0.158174 |              0.000000 |
| training_prevalence         |           0.158173 | -0.000001 | 0.133165 |  0.491357 |            0.155553 |              8.071123 |
| logistic                    |           0.158117 | -0.000057 | 0.133386 |  0.528420 |            0.171996 |              7.918510 |
| weighted_ensemble_raw       |           0.410852 |  0.252678 | 0.208659 |  0.521567 |            0.167605 |            -44.045346 |
| unweighted_xgb_raw          |           0.154005 | -0.004169 | 0.138421 |  0.526583 |            0.172597 |              4.442721 |
| unweighted_xgb_sigmoid      |           0.158121 | -0.000053 | 0.133153 |  0.507274 |            0.170930 |              8.079445 |
| unweighted_rf_raw           |           0.158419 |  0.000245 | 0.134568 |  0.523721 |            0.171433 |              7.102708 |
| unweighted_rf_sigmoid       |           0.157661 | -0.000513 | 0.133376 |  0.488134 |            0.155305 |              7.925670 |
| unweighted_ensemble_raw     |           0.155771 | -0.002403 | 0.136109 |  0.527317 |            0.174157 |              6.038821 |
| unweighted_ensemble_sigmoid |           0.157952 | -0.000222 | 0.133202 |  0.500004 |            0.166414 |              8.045411 |

<!-- RESULTS_END -->

A gain over the outdated 5% assumption may simply correct the portfolio rate. Do not claim individual ranking lift, causal drivers or retention uplift from rate calibration. The full report provides candidate comparisons, paired bootstrap intervals, proxy cohorts and reconciled historical exposure.

## Reproduce locally

Use Windows CPython 3.12.10 for the verified environment. `requirements.txt` pins direct dependencies; `requirements-lock.txt` records the complete tested Windows distribution set. Other platforms require their own dependency resolution and verification; cross-platform execution has not been certified. Install into a fresh virtual environment and run from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m ipykernel install --prefix .venv --name python3
python -m src.pipeline run
python -m unittest discover -s tests -p "test_*.py" -v
python -m jupyter nbconvert --to notebook --execute notebooks/*.ipynb --inplace
python tests/verify_all_steps.py
```

For individual stages:

```powershell
python -m src.pipeline train
python -m src.pipeline evaluate
python -m src.pipeline score --data data/Case_Study_2_CLV_Predictive_Analytics.csv
python -m src.pipeline score --data new_unlabeled_snapshot.csv --output new_score_run/CLV_SCORED_OUTPUT.csv
```

Paths resolve from the package location; scoring/evaluation also accept `--run-dir`. Scoring supports absent targets or entirely missing targets and only transforms with saved state. New scoring paths must be fresh. The frozen feature reference date is 2026-10-01; advancing it requires a new training run and explicitly observed as-of data.

To review the delivered evidence without generating a new run, install the environment, execute the three notebooks, and run the tests and verifier; skip `src.pipeline run`. Retraining creates new run IDs. Before committing a retrained submission, update the two run exceptions in `.gitignore` and the run IDs in the readiness review. `.gitattributes` disables line-ending conversion so a Git checkout preserves the exact bytes covered by integrity hashes ([Git documentation](https://git-scm.com/docs/gitattributes)). Score pointers use portable relative paths.

## Evidence and scoring artifacts

Training writes `outputs/runs/<training_run_id>/`: keyed full-precision OOF predictions, inference bundle, manifest with account lists/folds/seeds/versions/input and artifact hashes, metrics, bootstrap intervals, cohorts and report. Evaluation verifies evidence integrity and fold membership. The all-step verifier is read-only and cannot retrain or overwrite a report.

The three executed notebooks were reviewed on 2026-10-11. Notebook 1 covers data validation, cleaning, target grain and signed financial quantities. Notebook 2 covers fold isolation, persisted feature state, individual XGB/RF comparisons, calibration, unlabeled inference and all four component explanation plots. Notebook 3 covers uncertainty, cohorts, historical exposure, account array-event summaries, capacity diagnostics, per-array horizon scenarios, scoring lineage/export checks, future-data contracts and the Snowflake implementation and acceptance plan. Each notebook identifies its saved training run and checks source/evidence integrity. `tools/build_notebooks.py` maintains their source; notebooks read saved artifacts without retraining or replacing validation evidence.

Scoring writes `outputs/score_runs/<score_run_id>/`: 25-column CSV, validated input snapshot, score manifest and full XGB contribution maps. The manifest identifies the training run, original-input/snapshot/bundle/output hashes, UTC score timestamp, fixed reference date and policy. Top-level CSV/report files are convenience copies. Production scores include full-fit labeled rows and must never be used as validation.

Its `submission_analytics/` directory adds full-fit and grouped-OOF account array-event summaries, current-load capacity diagnostics, 15-year survival/event scenarios, discounted cash-flow scenarios and a separate hash/lineage manifest. Account probabilities are bounds on any/all array events conditional on estimated marginals; they are not cancellation probabilities. Missing array coverage withholds bounds, and missing labels withhold observed proxies. Unit annual cash-flow scenarios have no dollar units. Pass `--cash-flows` with all keyed years 1-15 to evaluate explicit forward USD cash flows, or `--product-mapping` with a genuine product-family lookup to add XGB/RF/ensemble product-cohort comparisons against both baselines.

```powershell
python -m src.pipeline score --cash-flows future_cash_flows.csv --product-mapping product_family_mapping.csv
```

These optional files are not supplied in the case-study dataset. Header-only input schemas are under `data/contracts/`; they contain no invented observations. [Business definitions and input contracts](docs/REMAINING_DATA_CONTRACTS.md) document the semantics, formulas and restrictions. `python -m src.temporal_validation --help` describes the dated workflow for array events, account cancellation or capacity expansion. It requires explicit observation/availability dates and an outcome definition, enforces matured training labels, fits preprocessing at each cutoff and calibrates on grouped training OOF predictions. It has regression coverage using temporary synthetic fixtures; no historical, account-cancellation or learned expansion performance is claimed on the supplied CSV.

The exported XGB and RF probabilities are individually sigmoid-mapped. The ensemble probability is a sigmoid mapping of the **raw** weighted mixture, rather than a mixture of the mapped component probabilities.

## Explanation and bucket contract

SHAP values explain the unweighted XGB component in **raw-margin log-odds units**. Contributions plus base reconstruct its margin; sigmoid(margin) reconstructs its raw probability. They do not add to the calibrated ensemble probability and are model associations, not causal effects. Full maps/base/model lineage are in the explanation sidecar; the CSV retains top three display columns.

The single bucket policy is CRITICAL >=0.70, HIGH >=0.40, MEDIUM >=0.15, LOW <0.15. Buckets are descriptive policies without demonstrated treatment benefit. Negative historical profit remains signed; avoiding a loss-making stream cannot be called losing positive revenue.

## Twelve-month score versus fifteen-year scenarios

`src.scenarios` implements assumed constant annual churn: S(t)=(1-p)^t. Discounted scenario value is sum of user-supplied annual cash flows times survival, divided by (1+r)^t, with payments at year end. No historical-profit-to-future-cash-flow conversion is made. At p=0 survival stays 1; at p=1 end-of-year survival is 0; at p=.05 S(15)=.463291. Years 2-15 are assumptions, not learned renewal forecasts. Event durations/censoring, retirement rules, discount rate and forward cash flows are needed for survival/CLV modeling.

## Snowflake handover

Run `sql/setup.sql`, `sql/ddl_clv_scored_output.sql`, then `sql/stored_proc_scoring_pipeline.sql` in a fresh selected sandbox database/schema/warehouse. Setup uses an internal fixed CLV_STAGE and exact CSV file format. The warehouse procedure uses a Python/Snowpark handler, a validated relative filename, per-call temporary staging, duplicate/version/probability/bucket checks, immutable run history and a run ledger. **CLV_SCORED_OUTPUT is a physical table**, maintained with history and the ledger in one transaction. Older score timestamps cannot overwrite newer scores; timestamp ties use lexical RUN_ID ordering. The independent `CLV_SCORED_LATEST_FROM_HISTORY` view supports reconciliation. Standard-table keys are informational, so **serialize ingestion through one writer job**, especially for retries of the same run ID. Same-run concurrent idempotency is not claimed. Existing deployments of the old view require a planned rename and history backfill; this project has not run a live migration.

Stage the CSV with SnowSQL (replace the example path with the absolute file path):
```sql
PUT file://C:/path/to/score_run/CLV_SCORED_OUTPUT.csv @CLV_STAGE/<score_run_id> AUTO_COMPRESS=FALSE OVERWRITE=FALSE;
CALL SP_INGEST_CLV_SCORED_BATCH(
  '<score_run_id>/CLV_SCORED_OUTPUT.csv', 'v2.1.0-grouped-sigmoid-unweighted',
  '<score_run_id>', '<training_run_id>', '<snapshot_sha256>', '<bundle_sha256>'
);
```

Use the exact values from `score_manifest.json`, and a unique stage filename/path per scoring run. The six arguments and fixed stage are intentional. A replay compares complete payloads and lineage; changed files under the same run ID fail. Never re-use a run ID for a new snapshot. Scalar top-three SHAP columns are XGB explanations; full contribution sidecars are retained locally and not automatically loaded.

Live compilation/execution remains unverified because no connected Snowflake environment was supplied. Run `sql/sandbox_acceptance.sql` and retain COPY, history MERGE and current-table MERGE query IDs before deployment. The acceptance plan covers replay, invalid/changed batches, older/tied timestamps, rollback and current/history reconciliation. Local fake-session tests verify Python handler control flow, not warehouse SQL execution. [COPY syntax](https://docs.snowflake.com/en/sql-reference/sql/copy-into-table), [standard-table constraints](https://docs.snowflake.com/en/sql-reference/constraints-overview), and [MERGE duplicate behavior](https://docs.snowflake.com/en/sql-reference/sql/merge) informed this design.

## Monitoring and rollback

Before scoring, validate schemas and artifact hashes. For each matured future cohort, monitor bias, Brier/log loss, calibration and ranking against the learned-rate baseline. Track category drift, utilization anomalies and profit/capacity distributions. Pause actionable targeting when evidence is weak. Keep run history and choose an earlier immutable training bundle with `--run-dir` for local rollback; warehouse consumers can filter an approved MODEL_VERSION rather than deleting history.

Additional required data: array/account event definitions, observation dates and maturity policy, product-family mapping, future expansion events and dated cash flows. No durations, product labels or cancellation rules are invented.

