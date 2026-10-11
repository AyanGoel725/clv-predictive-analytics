-- Execute only in a sandbox after setting role/warehouse/database/schema.
-- Record all query IDs and procedure results. These instructions are not evidence of execution.
-- 1. Stage a unique fixture file; call the six-argument procedure with score_manifest.json values.
-- 2. Repeat identical call: expect REPLAY_UNCHANGED and unchanged history counts/payload.
-- 3. Stage changed scores under the same RUN_ID: expect rejection.
-- 4. New RUN_ID with later scores: expect new history rows and current table updated.
-- 5. Stage duplicate keys, wrong version, invalid probability or bucket: expect rejection.
-- 6. Simulate failed COPY: history/ledger must remain unchanged; inspect failure log.
-- 7. Enforce one writer job; do not concurrently retry the same RUN_ID.
-- 8. Load an older timestamp: history grows, current-score rows must not regress.
-- 9. Fail the current-table MERGE: history/current/ledger transaction must roll back.
-- 10. Check timestamp ties: lexically larger RUN_ID wins in both table and history view.
SELECT RUN_ID, COUNT(*) AS ROWS, COUNT(DISTINCT ACCOUNT_ID || ':' || ARRAY_ID || ':' || MODEL_VERSION) AS UNIQUE_KEYS
FROM CLV_SCORED_HISTORY GROUP BY RUN_ID;
SELECT RUN_ID, ACCOUNT_ID, ARRAY_ID, MODEL_VERSION, COUNT(*) AS N
FROM CLV_SCORED_HISTORY GROUP BY 1,2,3,4 HAVING COUNT(*) > 1;
SELECT RUN_ID, COUNT(*) AS N FROM CLV_SCORE_RUNS GROUP BY 1 HAVING COUNT(*) > 1;
SELECT ACCOUNT_ID, ARRAY_ID, MODEL_VERSION, COUNT(*) AS N
FROM CLV_SCORED_OUTPUT GROUP BY 1,2,3 HAVING COUNT(*) > 1;
-- Expect zero rows; exclude the load timestamp from business-payload comparison.
SELECT * EXCLUDE LOADED_AT FROM CLV_SCORED_OUTPUT
MINUS SELECT * EXCLUDE LOADED_AT FROM CLV_SCORED_LATEST_FROM_HISTORY;
SELECT * EXCLUDE LOADED_AT FROM CLV_SCORED_LATEST_FROM_HISTORY
MINUS SELECT * EXCLUDE LOADED_AT FROM CLV_SCORED_OUTPUT;
SELECT * FROM CLV_SCORE_RUNS ORDER BY COMPLETED_AT DESC;
SELECT * FROM CLV_SCORE_FAILURES ORDER BY FAILED_AT DESC;
SELECT MODEL_VERSION, CHURN_RISK_BUCKET, COUNT(*) AS ARRAYS,
       AVG(ENSEMBLE_CHURN_PROBABILITY) AS MEAN_PROBABILITY,
       SUM(HISTORICAL_15YR_NET_PROFIT_USD) AS SIGNED_HISTORICAL_EXPOSURE_USD
FROM CLV_SCORED_OUTPUT GROUP BY 1,2;

