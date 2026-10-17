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
# 2. INDEX COLUMN ANALYSIS

print("\n========== INDEX COLUMN ANALYSIS ==========")

index_col = next(
    (name for name in ["_c0", "Unnamed: 0"] if name in df.columns),
    None,
)

if index_col is not None:
    index_stats = df.select(
        min(col(index_col)).alias("min_index"),
        max(col(index_col)).alias("max_index"),
        count(col(index_col)).alias("row_count"),
        sum(col(index_col).cast("double")).alias("actual_sum"),
    ).first()

    row_count = index_stats["row_count"]
    min_index = index_stats["min_index"]
    max_index = index_stats["max_index"]
    actual_sum = index_stats["actual_sum"]

    print("Index column:", index_col)
    print("Minimum index:", min_index)
    print("Maximum index:", max_index)
    print("Non-null index rows:", row_count)
    print("Actual sum:", actual_sum)

    if (
        row_count > 0
        and min_index is not None
        and max_index is not None
        and actual_sum is not None
    ):
        expected_sum = row_count * (min_index + max_index) / 2

        print("Expected arithmetic-sequence sum:", expected_sum)

        if actual_sum == expected_sum:
            print("Index sum matches the expected arithmetic-sequence sum.")
        else:
            print("Index sum does not match the expected arithmetic-sequence sum.")
else:
    print("No index column found. Skipping index analysis.")
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

display_columns = [
    "timestamp",
    "TP2",
    "TP3",
    "Oil_temperature",
    "Motor_current",
]

if index_col is not None:
    display_columns.insert(1, index_col)

first_rows = df.orderBy("timestamp").select(*display_columns).limit(10)
first_rows.show(truncate=False)


print("\n========== LAST ROWS ==========")

last_rows = df.orderBy(col("timestamp").desc()).select(*display_columns).limit(10)
last_rows.show(truncate=False)


print("\n========== TIME-SERIES ANALYSIS COMPLETE ==========")

spark.stop()
