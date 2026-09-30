from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    min,
    max,
    avg,
    stddev,
    count,
    sum,
    lag,
    when,
)
import os

from pyspark.sql.window import Window

spark = SparkSession.builder.appName("VehicleFaultTimeSeriesAnalysis").getOrCreate()

input_path = os.environ["PARQUET_PATH"]

df = spark.read.parquet(input_path)


# ============================================
# 1. DATA OVERVIEW
# ============================================

print("\n========== DATA OVERVIEW ==========")

print("Rows:", df.count())
print("Columns:", len(df.columns))

print("Start time:")
df.select(min("timestamp")).show()

print("End time:")
df.select(max("timestamp")).show()


# ============================================
# 2. UNDERSTAND _c0
# ============================================

print("\n========== _c0 ANALYSIS ==========")

index_stats = df.select(
    min("_c0").alias("min_index"),
    max("_c0").alias("max_index"),
    count("_c0").alias("row_count"),
    sum(col("_c0")).alias("actual_sum"),
).first()

row_count = index_stats["row_count"]
min_index = index_stats["min_index"]
max_index = index_stats["max_index"]
actual_sum = index_stats["actual_sum"]

expected_sum = row_count * (min_index + max_index) / 2

print("Min _c0:", min_index)
print("Max _c0:", max_index)
print("Rows:", row_count)
print("Actual sum:", actual_sum)
print("Expected arithmetic-sequence sum:", expected_sum)

if actual_sum == expected_sum:
    print("_c0 follows an arithmetic progression " "between its minimum and maximum.")
else:
    print("_c0 does not follow a simple arithmetic progression.")


# ============================================
# 3. TIMESTAMP GAP ANALYSIS
# ============================================

print("\n========== TIMESTAMP GAP ANALYSIS ==========")

timestamp_window = Window.orderBy("timestamp")

df_with_previous = df.withColumn(
    "previous_timestamp", lag("timestamp").over(timestamp_window)
)

df_with_gap = df_with_previous.withColumn(
    "gap_seconds",
    (col("timestamp").cast("long") - col("previous_timestamp").cast("long")),
)

gap_stats = df_with_gap.select(
    min("gap_seconds").alias("min_gap_seconds"),
    max("gap_seconds").alias("max_gap_seconds"),
    avg("gap_seconds").alias("avg_gap_seconds"),
    count(when(col("gap_seconds") != 10, True)).alias("non_10_second_gaps"),
).first()

print("Minimum gap:", gap_stats["min_gap_seconds"], "seconds")
print("Maximum gap:", gap_stats["max_gap_seconds"], "seconds")
print("Average gap:", gap_stats["avg_gap_seconds"], "seconds")
print("Non-10-second gaps:", gap_stats["non_10_second_gaps"])


# ============================================
# 4. SENSOR STATISTICS
# ============================================

print("\n========== SENSOR STATISTICS ==========")

sensor_columns = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "Caudal_impulses",
]

sensor_stats = []

for sensor in sensor_columns:
    sensor_stats.extend(
        [
            min(col(sensor)).alias(f"{sensor}_min"),
            max(col(sensor)).alias(f"{sensor}_max"),
            avg(col(sensor)).alias(f"{sensor}_mean"),
            stddev(col(sensor)).alias(f"{sensor}_stddev"),
        ]
    )

df.select(sensor_stats).show(truncate=False)


# ============================================
# 5. BINARY STATE DISTRIBUTION
# ============================================

print("\n========== OPERATING STATE DISTRIBUTION ==========")

flag_columns = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
]

for column in flag_columns:

    print(f"\n--- {column} ---")

    df.groupBy(column).count().orderBy(column).show()


# ============================================
# 6. DATA ORDER CHECK
# ============================================

print("\n========== TIMESTAMP ORDER CHECK ==========")

first_rows = (
    df.orderBy("timestamp")
    .select(
        "timestamp",
        "_c0",
        "TP2",
        "TP3",
        "Oil_temperature",
        "Motor_current",
    )
    .limit(10)
)

first_rows.show(truncate=False)


print("\n========== LAST ROWS ==========")

last_rows = (
    df.orderBy(col("timestamp").desc())
    .select(
        "timestamp",
        "_c0",
        "TP2",
        "TP3",
        "Oil_temperature",
        "Motor_current",
    )
    .limit(10)
)

last_rows.show(truncate=False)


print("\n========== TIME-SERIES ANALYSIS COMPLETE ==========")

spark.stop()
