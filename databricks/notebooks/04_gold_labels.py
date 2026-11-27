# Databricks notebook source

from pyspark.sql import functions as F

# ============================================================
# 1. TABLE CONFIGURATION
# ============================================================

INPUT_TABLE = "vehicle_fault_catalog.vehicle_fault_prediction.silver_timeseries"

OUTPUT_TABLE = "vehicle_fault_catalog.vehicle_fault_prediction.gold_labeled_sensor"


# ============================================================
# 2. DOCUMENTED FAILURE WINDOWS
# ============================================================

failure_windows = [
    ("F1", "2020-04-18 00:00:00", "2020-04-18 23:59:59"),
    ("F2", "2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("F3", "2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("F4", "2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]


# ============================================================
# 3. READ TIME-SERIES DATA
# ============================================================

print("\n========== READING TIME-SERIES DATA ==========")

df = spark.table(INPUT_TABLE)

print("Input table:", INPUT_TABLE)
print("Rows:", df.count())


# ============================================================
# 4. VALIDATE TIMESTAMP
# ============================================================

print("\n========== TIMESTAMP VALIDATION ==========")

if "timestamp" not in df.columns:
    raise ValueError("Required timestamp column is missing.")

null_timestamps = df.filter(F.col("timestamp").isNull()).count()

if null_timestamps > 0:
    raise ValueError(
        f"Found {null_timestamps} null timestamps. "
        "Resolve them before target creation."
    )


# ============================================================
# 5. CREATE FAULT LABEL
# ============================================================

print("\n========== CREATING FAULT LABEL ==========")

fault_condition = F.lit(False)

for failure_id, start_time, end_time in failure_windows:

    condition = (F.col("timestamp") >= F.lit(start_time).cast("timestamp")) & (
        F.col("timestamp") <= F.lit(end_time).cast("timestamp")
    )

    fault_condition = fault_condition | condition

df = df.withColumn("fault_label", F.when(fault_condition, F.lit(1)).otherwise(F.lit(0)))


# ============================================================
# 6. CREATE FAILURE ID
# ============================================================

print("\n========== CREATING FAILURE ID ==========")

failure_id_expression = F.lit("NO_FAILURE")

for failure_id, start_time, end_time in failure_windows:

    condition = (F.col("timestamp") >= F.lit(start_time).cast("timestamp")) & (
        F.col("timestamp") <= F.lit(end_time).cast("timestamp")
    )

    failure_id_expression = F.when(condition, F.lit(failure_id)).otherwise(
        failure_id_expression
    )

df = df.withColumn("failure_id", failure_id_expression)


# ============================================================
# 7. TARGET DISTRIBUTION
# ============================================================

print("\n========== TARGET DISTRIBUTION ==========")

df.groupBy("fault_label").count().orderBy("fault_label").show()


# ============================================================
# 8. FAILURE DISTRIBUTION
# ============================================================

print("\n========== FAILURE DISTRIBUTION ==========")

df.groupBy("failure_id").count().orderBy("failure_id").show()


# ============================================================
# 9. FAILURE PERIOD SUMMARY
# ============================================================

print("\n========== FAILURE PERIOD SUMMARY ==========")

df.filter(F.col("fault_label") == 1).groupBy("failure_id").agg(
    F.count("*").alias("fault_rows"),
    F.min("timestamp").alias("failure_start"),
    F.max("timestamp").alias("failure_end"),
).orderBy("failure_id").show(truncate=False)


# ============================================================
# 10. TARGET PERCENTAGE
# ============================================================

print("\n========== TARGET PERCENTAGE ==========")

target_stats = df.agg(
    F.count("*").alias("total_rows"), F.sum("fault_label").alias("fault_rows")
).first()

total_rows = target_stats["total_rows"]
fault_rows = int(target_stats["fault_rows"] or 0)

fault_percentage = fault_rows / total_rows * 100 if total_rows > 0 else 0

print("Total rows:", total_rows)
print("Fault rows:", fault_rows)
print(f"Fault percentage: {fault_percentage:.4f}%")


# ============================================================
# 11. SAVE GOLD LABELED TABLE
# ============================================================

print("\n========== SAVING GOLD LABELED TABLE ==========")

(
    df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE)
)


# ============================================================
# 12. VERIFY OUTPUT
# ============================================================

saved_df = spark.table(OUTPUT_TABLE)

print("Output table:", OUTPUT_TABLE)
print("Saved rows:", saved_df.count())

print("\n========== GOLD LABEL CREATION COMPLETE ==========")
