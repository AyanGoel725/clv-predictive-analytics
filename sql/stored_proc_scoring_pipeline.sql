-- Static design only until compiled and exercised in the user's Snowflake sandbox.
-- Per-call staging is isolated; the ingestion scheduler MUST serialize calls.
-- In particular, simultaneous retries of the same RUN_ID are not safe under informational keys.
CREATE OR REPLACE PROCEDURE SP_INGEST_CLV_SCORED_BATCH(
    FILE_NAME VARCHAR, TARGET_MODEL_VERSION VARCHAR, RUN_ID VARCHAR,
    TRAINING_RUN_ID VARCHAR, SNAPSHOT_SHA256 VARCHAR, BUNDLE_SHA256 VARCHAR
)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'ingest'
EXECUTE AS CALLER
AS
$$
import re
import uuid

COLUMNS = ["ACCOUNT_ID","ARRAY_ID","CONTRACT_END_DATE","HISTORICAL_15YR_NET_PROFIT_USD","CURRENT_CAPACITY_TB","UTILIZATION_RATE_PCT","RAW_UTILIZATION_RATE_PCT","UTILIZATION_CAPPED","IS_OVER_UTILIZED_RAW","IS_NEGATIVE_PROFIT","EXPANSION_POTENTIAL_SIGNAL","EXPANSION_SIGNAL_ENCODED","STATIC_BASELINE_CHURN_RATE","XGB_CHURN_PROBABILITY","RF_CHURN_PROBABILITY","ENSEMBLE_CHURN_PROBABILITY","CHURN_RISK_BUCKET","SHAP_TOP_FEATURE_1","SHAP_TOP_FEATURE_1_VALUE","SHAP_TOP_FEATURE_2","SHAP_TOP_FEATURE_2_VALUE","SHAP_TOP_FEATURE_3","SHAP_TOP_FEATURE_3_VALUE","SCORED_AT","MODEL_VERSION"]

def ingest(session, file_name, target_model_version, run_id, training_run_id, snapshot_sha256, bundle_sha256):
    if not re.fullmatch(r"[A-Za-z0-9_./-]{1,500}", file_name or "") or ".." in file_name or file_name.startswith("/"):
        raise ValueError("Use a safe relative file name on the fixed CLV_STAGE")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,50}", target_model_version or ""):
        raise ValueError("Invalid model version")
    for value in (run_id, training_run_id):
        if not re.fullmatch(r"[a-f0-9]{32}", value or ""):
            raise ValueError("Run IDs must be 32 lowercase hex characters")
    for value in (snapshot_sha256, bundle_sha256):
        if not re.fullmatch(r"[a-f0-9]{64}", value or ""):
            raise ValueError("Hashes must be 64 lowercase hex characters")

    staging = "TMP_CLV_" + uuid.uuid4().hex.upper()
    columns = ", ".join(COLUMNS)
    in_transaction = False
    try:
        session.sql("CREATE TEMPORARY TABLE " + staging + " LIKE CLV_SCORE_STAGING_TEMPLATE").collect()
        copy_sql = ("COPY INTO " + staging + " FROM @CLV_STAGE FILES = ('" + file_name +
                    "') FILE_FORMAT = (FORMAT_NAME = 'CLV_SCORE_CSV') ON_ERROR = 'ABORT_STATEMENT' FORCE = TRUE")
        copy_result = session.sql(copy_sql).collect()
        copy_query_id = session.sql("SELECT LAST_QUERY_ID()").collect()[0][0]
        rows = session.sql("SELECT COUNT(*) FROM " + staging).collect()[0][0]
        if rows == 0:
            raise ValueError("Empty or missing staged batch")
        duplicate_count = session.sql("SELECT COUNT(*) FROM (SELECT ACCOUNT_ID, ARRAY_ID, MODEL_VERSION FROM " +
            staging + " GROUP BY 1,2,3 HAVING COUNT(*) > 1)").collect()[0][0]
        if duplicate_count:
            raise ValueError("Duplicate scoring keys")
        invalid_sql = """
          SELECT COUNT(*) FROM {table}
          WHERE MODEL_VERSION <> ? OR
          CURRENT_CAPACITY_TB <= 0 OR UTILIZATION_RATE_PCT < 0 OR RAW_UTILIZATION_RATE_PCT < 0 OR
          UTILIZATION_CAPPED < 0 OR UTILIZATION_CAPPED > 1 OR
          ABS(UTILIZATION_RATE_PCT - RAW_UTILIZATION_RATE_PCT) > 0.000001 OR
          ABS(UTILIZATION_CAPPED - LEAST(RAW_UTILIZATION_RATE_PCT, 1)) > 0.000001 OR
          IS_OVER_UTILIZED_RAW <> (RAW_UTILIZATION_RATE_PCT > 1) OR
          IS_NEGATIVE_PROFIT <> (HISTORICAL_15YR_NET_PROFIT_USD < 0) OR
          ABS(STATIC_BASELINE_CHURN_RATE - 0.05) > 0.000001 OR
          XGB_CHURN_PROBABILITY < 0 OR XGB_CHURN_PROBABILITY > 1 OR
          RF_CHURN_PROBABILITY < 0 OR RF_CHURN_PROBABILITY > 1 OR
          ENSEMBLE_CHURN_PROBABILITY < 0 OR ENSEMBLE_CHURN_PROBABILITY > 1 OR
          EXPANSION_POTENTIAL_SIGNAL NOT IN ('UNKNOWN','Low','Medium','High') OR
          EXPANSION_SIGNAL_ENCODED <> CASE EXPANSION_POTENTIAL_SIGNAL
            WHEN 'UNKNOWN' THEN 0 WHEN 'Low' THEN 1 WHEN 'Medium' THEN 2 WHEN 'High' THEN 3 END OR
          CHURN_RISK_BUCKET <> CASE
            WHEN ENSEMBLE_CHURN_PROBABILITY >= 0.70 THEN 'CRITICAL'
            WHEN ENSEMBLE_CHURN_PROBABILITY >= 0.40 THEN 'HIGH'
            WHEN ENSEMBLE_CHURN_PROBABILITY >= 0.15 THEN 'MEDIUM' ELSE 'LOW' END
        """.format(table=staging)
        if session.sql(invalid_sql, params=[target_model_version]).collect()[0][0]:
            raise ValueError("Batch failed model/feature/probability/bucket checks")
        existing = session.sql(
            "SELECT TRAINING_RUN_ID, MODEL_VERSION, FILE_NAME, SNAPSHOT_SHA256, BUNDLE_SHA256, ROWS_LOADED "
            "FROM CLV_SCORE_RUNS WHERE RUN_ID = ?", params=[run_id]).collect()
        if existing:
            if len(existing) != 1 or tuple(existing[0]) != (
                    training_run_id, target_model_version, file_name, snapshot_sha256, bundle_sha256, rows):
                raise ValueError("Run ID reuse with changed lineage or duplicate ledger entries")
            difference = session.sql(
                "SELECT COUNT(*) FROM ((SELECT " + columns + " FROM " + staging +
                " MINUS SELECT " + columns + " FROM CLV_SCORED_HISTORY WHERE RUN_ID = ?) UNION ALL "
                "(SELECT " + columns + " FROM CLV_SCORED_HISTORY WHERE RUN_ID = ? MINUS SELECT " +
                columns + " FROM " + staging + "))", params=[run_id, run_id]).collect()[0][0]
            history_count = session.sql("SELECT COUNT(*) FROM CLV_SCORED_HISTORY WHERE RUN_ID = ?", params=[run_id]).collect()[0][0]
            if difference or history_count != rows:
                raise ValueError("Run ID replay contains changed scores or inconsistent history")
            return {"status": "REPLAY_UNCHANGED", "run_id": run_id, "rows": rows, "copy_query_id": copy_query_id}
        session.sql("BEGIN TRANSACTION").collect()
        in_transaction = True
        source = "SELECT " + columns + ", ? AS RUN_ID, ? AS TRAINING_RUN_ID FROM " + staging
        merge = (
            "MERGE INTO CLV_SCORED_HISTORY AS t USING (" + source + ") AS s "
            "ON t.ACCOUNT_ID=s.ACCOUNT_ID AND t.ARRAY_ID=s.ARRAY_ID AND "
            "t.MODEL_VERSION=s.MODEL_VERSION AND t.RUN_ID=s.RUN_ID "
            "WHEN NOT MATCHED THEN INSERT (" + columns + ", RUN_ID, TRAINING_RUN_ID) VALUES (" +
            ", ".join("s." + c for c in COLUMNS) + ", s.RUN_ID, s.TRAINING_RUN_ID)")
        merge_result = session.sql(merge, params=[run_id, training_run_id]).collect()
        merge_query_id = session.sql("SELECT LAST_QUERY_ID()").collect()[0][0]
        inserted = int(merge_result[0][0])
        if inserted != rows:
            raise ValueError("Partial/duplicate run insertion; inspect serialized ingestion")
        # Timestamp ties use RUN_ID consistently with the independent history view.
        latest_merge = (
            "MERGE INTO CLV_SCORED_OUTPUT AS t USING (" + source + ") AS s "
            "ON t.ACCOUNT_ID=s.ACCOUNT_ID AND t.ARRAY_ID=s.ARRAY_ID AND t.MODEL_VERSION=s.MODEL_VERSION "
            "WHEN MATCHED AND (s.SCORED_AT > t.SCORED_AT OR "
            "(s.SCORED_AT = t.SCORED_AT AND s.RUN_ID > t.RUN_ID)) THEN UPDATE SET " +
            ", ".join("t." + c + "=s." + c for c in COLUMNS + ["RUN_ID", "TRAINING_RUN_ID"]) +
            ", t.LOADED_AT=CURRENT_TIMESTAMP() "
            "WHEN NOT MATCHED THEN INSERT (" + columns + ", RUN_ID, TRAINING_RUN_ID) VALUES (" +
            ", ".join("s." + c for c in COLUMNS) + ", s.RUN_ID, s.TRAINING_RUN_ID)")
        session.sql(latest_merge, params=[run_id, training_run_id]).collect()
        latest_merge_query_id = session.sql("SELECT LAST_QUERY_ID()").collect()[0][0]
        session.sql(
            "INSERT INTO CLV_SCORE_RUNS (RUN_ID, TRAINING_RUN_ID, MODEL_VERSION, FILE_NAME, "
            "SNAPSHOT_SHA256, BUNDLE_SHA256, ROWS_LOADED, COPY_QUERY_ID, MERGE_QUERY_ID, LATEST_MERGE_QUERY_ID) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params=[run_id, training_run_id, target_model_version, file_name, snapshot_sha256,
                    bundle_sha256, rows, copy_query_id, merge_query_id, latest_merge_query_id]).collect()
        session.sql("COMMIT").collect()
        in_transaction = False
        return {"status": "SUCCESS", "run_id": run_id, "rows_loaded": rows,
                "rows_inserted": inserted, "copy_query_id": copy_query_id, "merge_query_id": merge_query_id,
                "latest_merge_query_id": latest_merge_query_id}
    except Exception as error:
        if in_transaction:
            session.sql("ROLLBACK").collect()
        session.sql("INSERT INTO CLV_SCORE_FAILURES (RUN_ID, ERROR_MESSAGE) VALUES (?, ?)",
                    params=[run_id, str(error)[:2000]]).collect()
        raise
    finally:
        session.sql("DROP TABLE IF EXISTS " + staging).collect()
$$;

