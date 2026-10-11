# Business contracts and supported completion

The supplied dataset is a cross-sectional array snapshot with a binary 12-month
array-event label. Code can implement other workflows, but it cannot supply missing
business outcomes. This document defines the inputs required by those workflows.
No additional business dataset has been fabricated. Synthetic data appears only
inside regression tests, whose temporary outputs are deleted after execution.

## Account outcomes

Current exports summarize array-event marginals per account. For array probabilities
`p_i`, the probability of any array event lies between `max(p_i)` and
`min(1, sum(p_i))`; the probability of all array events lies between
`max(0, sum(p_i) - n + 1)` and `min(p_i)`. These are bounds conditional on estimated
marginals. They make no independence assumption and are not account-cancellation
probabilities. Bounds are withheld when array probability coverage is incomplete.
Observed any/all-array event proxies are withheld unless every array has a label.

True cancellation modeling uses `account_cancelled_next_12m` supplied by the business
owner, with a written definition of cancellation, transfers, replacements and
partial renewals. Provide one dated feature snapshot per account, using the schema
in `data/contracts/dated_account_outcomes.csv`. Feature values must describe the
whole account under a declared aggregation policy: capacity/profit may be additive;
utilization must use a consistent account definition; contract date and expansion
tag must have explicit policies. No such policies are inferred from the array file.

## Chronological evaluation and expansion outcomes

`python -m src.temporal_validation` accepts real dated snapshots with:

| Column | Contract |
|---|---|
| account_id, array_id | Populated stable keys; array_id omitted for account snapshots |
| observation_date | Timestamp at which the forecast would have been made |
| feature_as_of | No later than observation_date; all covariates must genuinely be available then |
| label_window_end | Exactly 12 months after observation_date |
| label_available_at | Availability timestamp, no earlier than label_window_end for an observed label |
| outcome column | Binary 0/1 or missing; no label imputation |
| existing raw feature columns | Same units and validation as the main array pipeline |
| product_family (optional) | Actual product identity at observation; populated if supplied |

The outcome column is one of `churn_event_next_12m`,
`account_cancelled_next_12m`, or `capacity_expanded_next_12m`. The command requires
an explicit `--outcome-definition`. Expansion needs an actual realized capacity
change rule, including transfers/replacements and measurement dates. Its supplied
expansion tag is removed from the predictor to prevent a label-proxy shortcut;
the 5% churn baseline is not used as an expansion baseline.

For each cutoff, training uses labels whose windows and availability precede the
cutoff, keeping the latest eligible snapshot per entity. Calibration uses grouped
training-only OOF predictions. Validation uses matured outcomes from the cutoff
snapshot, evaluated at an explicit later `--evaluation-as-of`. Preprocessing uses
the cutoff as its reference date. Existing accounts can recur across time windows;
identifiers never enter model features. This evaluates later forecasts for an
existing portfolio, not exclusively new-account generalization. Results retain
window membership, bundles, metrics, versions, outcome definition and hashes.
Optional dated product families produce historical cohort metrics. Separate
temporal runs never replace the main grouped-OOF evidence or its latest pointer.

Example once real dated data is available:

```powershell
python -m src.temporal_validation --data dated_snapshots.csv --cutoffs 2022-01-01,2023-01-01 --evaluation-as-of 2025-01-01 --outcome-definition "Business-approved array-event definition"
```

The original case-study file is rejected by this command because the required dates
are absent. Test-fixture execution proves software behavior, not historical accuracy.

## Capacity signals available now

The delivered capacity file preserves the supplied categorical tag and reports:
`capacity * max(utilization - 1, 0)` as additional capacity to reach 100% at the
reported load, and `capacity * max(1 - utilization, 0)` as spare capacity at that load.
These are arithmetic diagnostics. Out-of-range utilization may reflect injected
data defects rather than real demand. Neither arithmetic nor the tag is a learned
future expansion probability or a purchase recommendation.

## Fifteen-year scenarios and cash flows

Every scored array receives years 1-15 of `(1-p)^t` under a constant annual-event-rate
assumption, compared with 5%. The binary array label does not establish that events
are absorbing losses or that its complement is commercial renewal. Therefore these
exports are named scenarios and never claim learned survival or renewal.

Without forward cash flows, discounted values use one unit annually at 0%, 5% and
10% sensitivity rates. They have no dollar units. To supply a dollar scenario, pass
`--cash-flows` with the exact snapshot account/array keys, every integer year 1-15,
and `annual_cash_flow_usd`. Missing/duplicate years and unknown keys fail validation;
negative cash flows remain signed. Historical accumulated profit is never converted
to future cash flow. Event durations/censoring and dated financial projections are
needed to validate a learned survival/CLV model.

## Actual product-family diagnostics

Pass `--product-mapping` with `account_id,array_id,product_family`. The lookup must
cover every evaluated array without duplicates or missing product labels. The
cross-sectional lookup assumes the supplied family applies to the evaluated
snapshot; use dated product identity for changing families in historical evaluation.
Diagnostics compare calibrated XGB, RF and ensemble against both 5% and training
prevalence, retaining account bootstrap intervals. No product families are guessed
from contract expiry or expansion tags. Without a lookup, the manifest explicitly
records `unavailable_no_product_mapping`.

## Warehouse execution

`CLV_SCORED_OUTPUT` is a physical current-score table. The procedure transaction
writes immutable history, current scores and the run ledger. Older scored timestamps
cannot replace newer ones; equal timestamps use lexical RUN_ID ordering. An
independent history-derived view reconciles the table. Execute the acceptance plan
in a fresh Snowflake sandbox and retain COPY, history MERGE and current MERGE query
IDs. Local fake-session tests check handler control flow only, not Snowflake SQL
compilation or execution. Enforce one writer job because standard-table keys do not
enforce uniqueness. Existing deployments of the earlier view need a planned rename
and history backfill before using the new physical table; no live migration is run.
