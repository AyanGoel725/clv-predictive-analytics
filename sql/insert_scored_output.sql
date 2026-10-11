-- Ingestion is performed by SP_INGEST_CLV_SCORED_BATCH, which validates before MERGE.
-- Never directly MERGE production scores without a unique run ID and checked staging.
-- Stage a unique <score_run_id>/CLV_SCORED_OUTPUT.csv using the README PUT command.
-- Then substitute exact score_manifest.json values into this six-argument insert call:
-- CALL SP_INGEST_CLV_SCORED_BATCH(
--   '<score_run_id>/CLV_SCORED_OUTPUT.csv', 'v2.1.0-grouped-sigmoid-unweighted',
--   '<score_run_id>', '<training_run_id>', '<snapshot_sha256>', '<bundle_sha256>'
-- );
-- The handler performs INSERT through MERGE into CLV_SCORED_OUTPUT and history,
-- then commits both tables and the ledger atomically. Retain all three query IDs.
-- Execute only in a sandbox after setting role/warehouse/database/schema.
-- Record all query IDs and procedure results. These instructions are not evidence of execution.
-- 1. Stage a unique fixture file; call the six-argument procedure with score_manifest.json values.
-- 2. Repeat identical call: expect REPLAY_UNCHANGED and unchanged history counts/payload.
-- 3. Stage changed scores under the same RUN_ID: expect rejection.
-- 4. New RUN_ID with later scores: expect new history rows and the current table updated.
-- 5. Stage duplicate keys, wrong version, invalid probability or bucket: expect rejection.
-- 6. Simulate failed COPY: history/ledger must remain unchanged; inspect failure log.
-- 7. Enforce one writer job; do not concurrently retry the same RUN_ID.
SELECT RUN_ID, COUNT(*) AS ROWS, COUNT(DISTINCT ACCOUNT_ID || ':' || ARRAY_ID || ':' || MODEL_VERSION) AS UNIQUE_KEYS
FROM CLV_SCORED_HISTORY GROUP BY RUN_ID;
SELECT RUN_ID, ACCOUNT_ID, ARRAY_ID, MODEL_VERSION, COUNT(*) AS N
FROM CLV_SCORED_HISTORY GROUP BY 1,2,3,4 HAVING COUNT(*) > 1;
SELECT RUN_ID, COUNT(*) AS N FROM CLV_SCORE_RUNS GROUP BY 1 HAVING COUNT(*) > 1;
SELECT * FROM CLV_SCORE_RUNS ORDER BY COMPLETED_AT DESC;
SELECT * FROM CLV_SCORE_FAILURES ORDER BY FAILED_AT DESC;
SELECT MODEL_VERSION, CHURN_RISK_BUCKET, COUNT(*) AS ARRAYS,
       AVG(ENSEMBLE_CHURN_PROBABILITY) AS MEAN_PROBABILITY,
       SUM(HISTORICAL_15YR_NET_PROFIT_USD) AS SIGNED_HISTORICAL_EXPOSURE_USD
FROM CLV_SCORED_OUTPUT GROUP BY 1,2;

