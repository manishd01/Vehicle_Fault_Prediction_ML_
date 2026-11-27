# Databricks notebook source

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
)
import uuid

# ============================================================
# 1. TABLE CONFIGURATION
# ============================================================

BRONZE_TABLE = "vehicle_fault_catalog.vehicle_fault_prediction.bronze_sensor_raw"

SILVER_TABLE = "vehicle_fault_catalog.vehicle_fault_prediction.silver_sensor_clean"

QUARANTINE_TABLE = (
    "vehicle_fault_catalog.vehicle_fault_prediction.quarantine_sensor_records"
)

QUALITY_TABLE = "vehicle_fault_catalog.vehicle_fault_prediction.data_quality_metrics"

run_id = str(uuid.uuid4())


# ============================================================
# 2. READ BRONZE DATA
# ============================================================

print("Reading Bronze Delta table...")

df = spark.table(BRONZE_TABLE)

total_rows = df.count()

print("Total rows:", total_rows)
print("Total columns:", len(df.columns))

df.printSchema()


# ============================================================
# 3. NULL CHECK FOR ALL COLUMNS
# ============================================================

print("\n========== NULL CHECK ==========")

null_expressions = [
    F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(f"null_{i}")
    for i, column in enumerate(df.columns)
]

null_stats = df.agg(*null_expressions).first()

metric_rows = [("total_rows", total_rows, "INFO", run_id)]

for i, column in enumerate(df.columns):
    null_count = int(null_stats[f"null_{i}"] or 0)

    print(f"{column}: {null_count} null values")

    metric_rows.append(
        (
            f"null_count_{column}",
            null_count,
            "PASS" if null_count == 0 else "WARNING",
            run_id,
        )
    )


# ============================================================
# 4. TIMESTAMP VALIDATION
# ============================================================

print("\n========== TIMESTAMP CHECK ==========")

null_timestamp_rows = df.filter(F.col("timestamp").isNull()).count()

print("Null timestamps:", null_timestamp_rows)

metric_rows.append(
    (
        "null_timestamp_rows",
        null_timestamp_rows,
        "PASS" if null_timestamp_rows == 0 else "WARNING",
        run_id,
    )
)


# ============================================================
# 5. DUPLICATE TIMESTAMP CHECK
# ============================================================

print("\n========== DUPLICATE TIMESTAMP CHECK ==========")

duplicate_timestamp_rows = (
    df.groupBy("timestamp")
    .count()
    .filter(F.col("count") > 1)
    .agg(F.coalesce(F.sum(F.col("count") - 1), F.lit(0)).alias("duplicate_rows"))
    .first()["duplicate_rows"]
)

duplicate_timestamp_rows = int(duplicate_timestamp_rows or 0)

print("Duplicate timestamp rows:", duplicate_timestamp_rows)

metric_rows.append(
    (
        "duplicate_timestamp_rows",
        duplicate_timestamp_rows,
        "PASS" if duplicate_timestamp_rows == 0 else "WARNING",
        run_id,
    )
)


# ============================================================
# 6. ORIGINAL INDEX CHECK
# ============================================================

print("\n========== INDEX CHECK ==========")

index_column = None

if "_c0" in df.columns:
    index_column = "_c0"
elif "Unnamed: 0" in df.columns:
    index_column = "Unnamed: 0"

if index_column:
    duplicate_index_rows = (
        df.groupBy(index_column)
        .count()
        .filter(F.col("count") > 1)
        .agg(F.coalesce(F.sum(F.col("count") - 1), F.lit(0)).alias("duplicate_rows"))
        .first()["duplicate_rows"]
    )

    duplicate_index_rows = int(duplicate_index_rows or 0)

    print("Index column:", index_column)
    print("Duplicate index rows:", duplicate_index_rows)

    metric_rows.append(
        (
            "duplicate_index_rows",
            duplicate_index_rows,
            "PASS" if duplicate_index_rows == 0 else "WARNING",
            run_id,
        )
    )


# ============================================================
# 7. CREATE QUARANTINE DATA
# ============================================================

print("\n========== QUARANTINE INVALID RECORDS ==========")

# For now, records with missing timestamps are quarantined.
# We are not dropping sensor outliers without domain validation.

quarantine_df = df.filter(F.col("timestamp").isNull())

quarantine_count = quarantine_df.count()

(
    quarantine_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(QUARANTINE_TABLE)
)

print("Quarantined records:", quarantine_count)


# ============================================================
# 8. CREATE SILVER CLEAN TABLE
# ============================================================

print("\n========== CREATE SILVER TABLE ==========")

silver_df = df.filter(F.col("timestamp").isNotNull())

(
    silver_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE)
)

print("Silver rows:", silver_df.count())
print("Silver table created:", SILVER_TABLE)


# ============================================================
# 9. PERSIST DATA QUALITY METRICS
# ============================================================

print("\n========== SAVE QUALITY METRICS ==========")

metrics_schema = StructType(
    [
        StructField("metric_name", StringType(), False),
        StructField("metric_value", LongType(), False),
        StructField("status", StringType(), False),
        StructField("run_id", StringType(), False),
    ]
)

metrics_df = spark.createDataFrame(metric_rows, schema=metrics_schema).withColumn(
    "checked_at", F.current_timestamp()
)

(metrics_df.write.format("delta").mode("append").saveAsTable(QUALITY_TABLE))

print("Quality metrics saved:", QUALITY_TABLE)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n========== SILVER QUALITY PIPELINE COMPLETE ==========")

print("Bronze table:", BRONZE_TABLE)
print("Silver table:", SILVER_TABLE)
print("Quarantine table:", QUARANTINE_TABLE)
print("Quality metrics table:", QUALITY_TABLE)
print("Run ID:", run_id)
