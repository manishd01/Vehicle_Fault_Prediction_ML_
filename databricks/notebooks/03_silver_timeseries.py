# Databricks notebook source

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ============================================================
# 1. TABLE CONFIGURATION
# ============================================================

INPUT_TABLE = "vehicle_fault_catalog.vehicle_fault_prediction.silver_sensor_clean"

OUTPUT_TABLE = "vehicle_fault_catalog.vehicle_fault_prediction.silver_timeseries"


# ============================================================
# 2. READ SILVER DATA
# ============================================================

print("\n========== LOADING SILVER DATA ==========")

df = spark.table(INPUT_TABLE)

total_rows = df.count()

print("Input table:", INPUT_TABLE)
print("Rows:", total_rows)
print("Columns:", len(df.columns))


# ============================================================
# 3. TIMESTAMP VALIDATION
# ============================================================

print("\n========== TIMESTAMP VALIDATION ==========")

null_timestamp_count = df.filter(F.col("timestamp").isNull()).count()

print("Null timestamps:", null_timestamp_count)

if null_timestamp_count > 0:
    raise ValueError(
        "Silver input contains null timestamps. "
        "Resolve these records before time-series preparation."
    )


# ============================================================
# 4. DUPLICATE TIMESTAMP CHECK
# ============================================================

print("\n========== DUPLICATE TIMESTAMP CHECK ==========")

duplicate_stats = (
    df.groupBy("timestamp")
    .count()
    .filter(F.col("count") > 1)
    .agg(F.coalesce(F.sum(F.col("count") - 1), F.lit(0)).alias("duplicate_rows"))
    .first()
)

duplicate_timestamp_rows = int(duplicate_stats["duplicate_rows"] or 0)

print("Duplicate timestamp rows:", duplicate_timestamp_rows)

if duplicate_timestamp_rows > 0:
    print("WARNING: Duplicate timestamps found.")
    df.groupBy("timestamp").count().filter(F.col("count") > 1).orderBy(
        "timestamp"
    ).show(20, truncate=False)


# ============================================================
# 5. CHRONOLOGICAL WINDOW
# ============================================================

print("\n========== CREATING TIME ORDER ==========")

# Use the original index as a tie-breaker if available.
# This provides deterministic ordering when timestamps match.

order_columns = [F.col("timestamp").asc()]

if "_c0" in df.columns:
    order_columns.append(F.col("_c0").asc())

time_window = Window.orderBy(*order_columns)


# ============================================================
# 6. CALCULATE TIME SINCE PREVIOUS READING
# ============================================================

print("\n========== CALCULATING TIME GAPS ==========")

df = df.withColumn("previous_timestamp", F.lag("timestamp").over(time_window))

df = df.withColumn(
    "time_since_previous_seconds",
    F.unix_timestamp("timestamp") - F.unix_timestamp("previous_timestamp"),
)

df = df.drop("previous_timestamp")


# ============================================================
# 7. TIME GAP SUMMARY
# ============================================================

print("\n========== TIME GAP SUMMARY ==========")

df.select(
    F.min("time_since_previous_seconds").alias("min_gap_seconds"),
    F.max("time_since_previous_seconds").alias("max_gap_seconds"),
    F.avg("time_since_previous_seconds").alias("avg_gap_seconds"),
).show(truncate=False)


# ============================================================
# 8. TIMESTAMP RANGE
# ============================================================

print("\n========== TIMESTAMP RANGE ==========")

df.select(
    F.min("timestamp").alias("start_time"),
    F.max("timestamp").alias("end_time"),
).show(truncate=False)


# ============================================================
# 9. VERIFY FIRST AND LAST RECORDS
# ============================================================

print("\n========== FIRST RECORDS ==========")

df.orderBy("timestamp").select("timestamp", "time_since_previous_seconds").show(
    10, truncate=False
)


print("\n========== LAST RECORDS ==========")

df.orderBy(F.col("timestamp").desc()).select(
    "timestamp", "time_since_previous_seconds"
).show(10, truncate=False)


# ============================================================
# 10. SAVE SILVER TIME-SERIES DELTA TABLE
# ============================================================

print("\n========== SAVING TIME-SERIES DATA ==========")

(
    df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE)
)


# ============================================================
# 11. VERIFY OUTPUT
# ============================================================

print("\n========== VERIFYING OUTPUT TABLE ==========")

saved_df = spark.table(OUTPUT_TABLE)

print("Output table:", OUTPUT_TABLE)
print("Saved rows:", saved_df.count())

saved_df.printSchema()

print("\n========== TIME-SERIES PREPARATION COMPLETE ==========")
