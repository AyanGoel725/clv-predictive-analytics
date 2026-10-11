# Submission readiness review - 11 October 2026

## Verdict

The final local review passed. The project is ready for preparing the final PDF,
provided the report explicitly describes the evidence limits below.
The supplied-data codebase, validation report, executed notebooks and warehouse
scripts are complete and locally checked. The full business mandate is only partly
empirically demonstrable from the supplied cross-sectional dataset. Account
cancellation, true historical validation, product-family results and learned
15-year/expansion outcomes must not be claimed without their corresponding data.
Snowflake execution remains unverified. The final PDF is still to be prepared.
The project is published at [AyanGoel725/clv-predictive-analytics](https://github.com/AyanGoel725/clv-predictive-analytics)
on `main`; record the exact published commit in the PDF.
This is a locally verified, data-limited case study,
not proof that every business outcome in the mandate has been demonstrated.

## Verified delivery

- Model version: `v2.1.0-grouped-sigmoid-unweighted`.
- Training run: `f0095e2660d447bd9df382f307062afb`.
- Scoring run: `a8f54120f11648ea87a4028cc6c98e95`.
- Full pipeline completed with 2,000 scored arrays.
- 43 regression tests passed, including integration, portable score-pointer and transaction-control tests.
- Three notebooks executed with zero cell errors; each identifies the current run.
- Read-only verifier passed input, training snapshot, source/SQL, OOF, bundle,
  scoring, derived output and report integrity checks.
- Dependency metadata matched the tested environment; `pip check` passed.
- A temporary Git index/checkout with `core.autocrlf=true` preserved all 67
  submission files byte for byte. The verifier and all 43 tests also passed from
  that checkout using the existing pinned Windows environment; this did not
  reinstall dependencies or publish a remote repository.
- Account delivery: 1,000 full-fit account summaries and 1,000 OOF coverage summaries.
  Of the OOF summaries, 889 accounts have complete array-label/prediction coverage;
  111 have incomplete coverage. Incomplete bounds and observed proxies are withheld.
- Capacity delivery: 2,000 current-load diagnostic records.
- Horizon delivery: 4,000 array/scenario records with years 1-15; 12,000 discounted
  sensitivity records at 0%, 5% and 10%, using one unit annually, without USD claims.

## Mandate and evidence

| Requirement | Implemented evidence | Remaining empirical limit |
|---|---|---|
| Two ML models | XGB/RF individual raw/calibrated metrics and account-bootstrap comparisons, ensemble and challengers | No demonstrated individual targeting advantage |
| Array-event prediction | Grouped nested-calibration evaluation and persisted inference | Event taxonomy and maturity of supplied labels are not externally verified |
| Account outcomes | Array-event summaries and dependence-free conditional probability bounds; dated true-account-target workflow | The array labels do not define commercial cancellation; no true account outcomes supplied |
| Fifteen-year CLV | Per-array constant-rate survival/event scenarios and explicit cash-flow interface | No learned long-term survival, verified renewal or supplied forward cash flows |
| Capacity expansion | Supplied tag plus headroom/pressure diagnostics; dated realized-expansion workflow excluding the tag as an outcome proxy | No observed future expansion outcomes supplied |
| Forecast bias | Proper OOF scores, signed improvement, calibration, both baselines and account uncertainty | Grouped cross-sectional CV is not historical back-testing |
| Historical validation | Rolling-origin workflow with as-of/maturity guards and per-window preprocessing/calibration; fixture tests | Original CSV lacks genuine observation and label-availability timestamps |
| Product cohorts | Actual lookup interface, both baselines, XGB/RF/ensemble diagnostics; dated families supported in temporal runs | No lookup provided; existing expiry-year/expansion cohorts remain proxies |
| Warehouse deliverables | Physical CLV_SCORED_OUTPUT, history/current/ledger transaction, replay/rollback and reconciliation scripts | No live sandbox compilation, load or acceptance query IDs |
| Technical handover | Executed notebooks, pinned environment, input contracts, hashes, test/verification commands, README and public GitHub repository | PDF and its exact revision link remain to be assembled |

## Quantitative conclusion

The observed labeled array-event rate is 15.8174%. Static 5% has Brier 0.144857.
The prespecified sigmoid ensemble has Brier 0.133202, an 8.05% improvement against
5%, and ROC-AUC 0.500004. Training-fold prevalence has Brier 0.133165. The paired
gain interval versus prevalence includes zero, so no reliable Brier advantage is
established. The supported finding is a portfolio-rate correction, not demonstrated
behavioral retention targeting or a selected production champion. Candidate
selection on these outer results would require nested selection or new outcomes.

## Cleanup and packaging review

Unused legacy metric/threshold-tuning, model-serialization and SQL-token helpers
were removed from `src/utils.py`; the active shared risk policy is retained.
The score pointer now uses forward-slash relative paths. `.gitattributes` preserves
the exact source and evidence bytes through Git line-ending settings. Only the
selected final training and scoring runs remain. Python caches, obsolete run
directories, earlier submissions and legacy notebook-generation/check scripts
are excluded or removed.

Keep the three notebooks, source, tests, notebook builder, both dependency files,
SQL scripts, original CSV, documented input templates and final output artifacts.
The top-level score/report copies and SHAP plots are used by the notebook workflow;
the run snapshots and bundle are needed for integrity checks and reproducibility.
The local `.venv` is retained for execution and ignored by Git. Do not upload it,
Python caches or notebook checkpoints. Header-only contract files are input
templates, not additional evidence or unfinished training data.

## Before submitting the PDF

1. Draft the PDF using the verified final run; distinguish completed implementation
   from absent empirical evidence and unverified warehouse execution.
2. Present individual XGB and RF results alongside the ensemble and both baselines.
   Include anomalies, split/calibration design, uncertainty, proxy cohort bias,
   SHAP units, account semantics, scenario assumptions and the warehouse architecture.
3. Run `sql/sandbox_acceptance.sql` and retain query IDs if sandbox access becomes
   available; otherwise disclose that live warehouse execution was not tested.
4. Use the public repository above and record the final commit SHA in the PDF.
   Verify the report refers to that exact published revision; the local Git
   checkout verification has already passed.

Missing business inputs and exact schemas are in `REMAINING_DATA_CONTRACTS.md`.
Providing fictitious longitudinal history, account cancellations or product labels
would invalidate the conclusions; test-fixture outcomes are not submission evidence.
