# Databricks notebook source

# ============================================================
# STAGE 01: BRONZE INGESTION
# Project: Vehicle Fault Prediction
#
# Purpose:
#   1. Read raw MetroPT3 CSV from Unity Catalog Volume.
#   2. Preserve original sensor columns.
#   3. Add ingestion metadata.
#   4. Write data into a managed Delta table.
#   5. Make repeat ingestion of the same file idempotent.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import hashlib
import uuid
from pathlib import Path
import re

from pyspark.sql import functions as F

# ============================================================
# 2. NOTEBOOK PARAMETERS
# ============================================================

# dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("catalog", "vehicle_fault_catalog")
dbutils.widgets.text("schema", "vehicle_fault_prediction")
dbutils.widgets.text("volume", "vehicle_fault_data")

dbutils.widgets.text("source_file", "MetroPT3(AirCompressor).csv")

dbutils.widgets.text("machine_id", "metropt3_compressor_01")

dbutils.widgets.text("table_name", "bronze_sensor_raw")


# Read parameters

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")

source_file = dbutils.widgets.get("source_file")
machine_id = dbutils.widgets.get("machine_id")
table_name = dbutils.widgets.get("table_name")


# ============================================================
# 3. VALIDATE IDENTIFIERS
# ============================================================


def validate_identifier(value, name):

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid {name}: {value}")

    return value


catalog = validate_identifier(catalog, "catalog")
schema = validate_identifier(schema, "schema")
volume = validate_identifier(volume, "volume")
table_name = validate_identifier(table_name, "table_name")


# ============================================================
# 4. BUILD UNITY CATALOG PATHS
# ============================================================

volume_path = f"/Volumes/{catalog}/{schema}/{volume}"

source_path = f"{volume_path}/{source_file}"

full_table_name = f"{catalog}.{schema}.{table_name}"


print("=" * 70)
print("VEHICLE FAULT PREDICTION - BRONZE INGESTION")
print("=" * 70)

print("Catalog:", catalog)
print("Schema:", schema)
print("Volume:", volume)
print("Source:", source_path)
print("Target table:", full_table_name)
print("Machine ID:", machine_id)


# ============================================================
# 5. VALIDATE SOURCE FILE
# ============================================================

input_file = Path(source_path)

if not input_file.is_file():

    raise FileNotFoundError(
        f"Source file not found: {source_path}\n"
        "Upload the MetroPT3 CSV into the configured "
        "Unity Catalog volume."
    )


# ============================================================
# 6. CALCULATE SOURCE FILE SHA256
# ============================================================
#
# A stable file hash helps identify the exact input file.
#
# If the same file is submitted again, the same transaction
# identifier is used, preventing duplicate Delta appends.
#
# If the contents change, a new hash is generated.
# ============================================================

sha256 = hashlib.sha256()

with open(source_path, "rb") as file:

    while True:

        chunk = file.read(8 * 1024 * 1024)

        if not chunk:
            break

        sha256.update(chunk)


source_hash = sha256.hexdigest()

print("Source SHA256:", source_hash)


# ============================================================
# 7. READ RAW CSV USING SPARK
# ============================================================
#
# We use Spark instead of pandas because the original dataset
# contains approximately 1.5 million records.
#
# Bronze preserves the source sensor columns without applying
# Silver cleaning or Gold feature transformations.
# ============================================================

df = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
    .option("mode", "PERMISSIVE")
    .load(source_path)
)


# ============================================================
# 8. BASIC INPUT VALIDATION
# ============================================================

if not df.columns:

    raise ValueError("The source CSV contains no columns.")


if len(df.columns) != len(set(df.columns)):

    raise ValueError("Duplicate column names detected in source CSV.")


if "timestamp" not in df.columns:

    raise ValueError("Required timestamp column is missing.")


if "machine_id" in df.columns:

    raise ValueError(
        "Source already contains machine_id. "
        "Review the source schema before adding metadata."
    )


row_count = df.count()

if row_count == 0:

    raise ValueError("Source dataset contains zero records.")


print("Input rows:", row_count)
print("Input columns:", len(df.columns))

print("\nOriginal source schema:")

df.printSchema()


# ============================================================
# 9. ADD BRONZE METADATA
# ============================================================
#
# Original columns are preserved.
#
# Additional metadata:
#   machine_id
#   _source_file
#   _source_file_sha256
#   _ingested_at
#   _run_id
# ============================================================

run_id = str(uuid.uuid4())

bronze_df = (
    df.withColumn("machine_id", F.lit(machine_id))
    .withColumn("_source_file", F.lit(source_file))
    .withColumn("_source_file_sha256", F.lit(source_hash))
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_run_id", F.lit(run_id))
)


# ============================================================
# 10. WRITE TO BRONZE DELTA TABLE
# ============================================================
#
# append:
#   Preserves previous ingestion history.
#
# txnAppId + txnVersion:
#   Prevent duplicate writes when the same file is retried.
#
# A new source file hash produces a new transaction ID.
# ============================================================

transaction_app_id = f"vehicle-fault-bronze-{machine_id}-{source_hash}"

print("\nWriting Bronze Delta table...")
print("Transaction App ID:", transaction_app_id)


(
    bronze_df.write.format("delta")
    .mode("append")
    .option("txnAppId", transaction_app_id)
    .option("txnVersion", 1)
    .saveAsTable(full_table_name)
)


# ============================================================
# 11. VERIFY DELTA TABLE
# ============================================================

print("\nBronze ingestion completed.")

print("Table:", full_table_name)
print("Source rows:", row_count)
print("Run ID:", run_id)


# Verify table provider and location

spark.sql(f"DESCRIBE DETAIL {full_table_name}").show(truncate=False)


# Inspect Delta table history

print("\nDelta table history:")

spark.sql(f"DESCRIBE HISTORY {full_table_name}").show(5, truncate=False)


# ============================================================
# 12. VERIFY SAMPLE RECORDS
# ============================================================

print("\nBronze table sample:")

display(
    spark.table(full_table_name)
    .select(
        "machine_id",
        "timestamp",
        "_source_file",
        "_source_file_sha256",
        "_ingested_at",
        "_run_id",
    )
    .limit(10)
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("=" * 70)
print("BRONZE INGESTION SUCCESSFUL")
print("=" * 70)

print("Source file:", source_file)
print("Source hash:", source_hash)
print("Input rows:", row_count)
print("Target table:", full_table_name)
print("Run ID:", run_id)

print("\nNext stage: 02_silver_quality.py")


# testcode:----------------------
# # Verify the uploaded MetroPT3 dataset

# source_path = "/Volumes/vehicle_fault_catalog/vehicle_fault_prediction/vehicle_fault_data/MetroPT3(AirCompressor).csv"

# import os

# print("File exists:", os.path.isfile(source_path))
# print("File size (bytes):", os.path.getsize(source_path))

# # Read first 5 lines
# with open(source_path, "r", encoding="utf-8-sig") as f:
#     for i in range(5):
#         print(f.readline().rstrip())
