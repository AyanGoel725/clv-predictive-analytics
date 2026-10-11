-- Select an existing sandbox role/warehouse/database/schema in your SQL client first.
-- Execute scripts in order: setup, ddl, stored procedure, sandbox_acceptance.
ALTER SESSION SET TIMEZONE = 'UTC';
CREATE FILE FORMAT IF NOT EXISTS CLV_SCORE_CSV
 TYPE = CSV COMPRESSION = AUTO SKIP_HEADER = 1
 FIELD_OPTIONALLY_ENCLOSED_BY = '"' NULL_IF = ('', 'NULL', 'null', 'NaN')
 DATE_FORMAT = 'YYYY-MM-DD' TIMESTAMP_FORMAT = 'YYYY-MM-DD HH24:MI:SS'
 ERROR_ON_COLUMN_COUNT_MISMATCH = TRUE;
CREATE STAGE IF NOT EXISTS CLV_STAGE FILE_FORMAT = CLV_SCORE_CSV;

