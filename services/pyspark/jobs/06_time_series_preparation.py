from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    min,
    max,
    lag,
    unix_timestamp,
)
from pyspark.sql.window import Window
import os

spark = SparkSession.builder.appName("VehicleFaultTimeSeriesPreparation").getOrCreate()

# ==============================
# PATH CONFIGURATION
# ==============================

data_root = os.environ["DATA_ROOT"]

input_path = os.environ["PARQUET_PATH"]
output_path = f"{data_root}/processed/metropt_timeseries"

# ============================================
# 1. LOAD DATA
# ============================================

print("\n========== LOADING DATA ==========")

df = spark.read.parquet(input_path)

print("Rows:", df.count())
print("Columns:", len(df.columns))


# ============================================
# 2. CHECK TIMESTAMP NULLS
# ============================================

print("\n========== TIMESTAMP VALIDATION ==========")

timestamp_nulls = df.filter(col("timestamp").isNull()).count()

print("Null timestamps:", timestamp_nulls)


# ============================================
# 3. CHECK DUPLICATE TIMESTAMPS
# ============================================

print("\n========== DUPLICATE TIMESTAMP CHECK ==========")

duplicate_timestamps = df.groupBy("timestamp").count().filter(col("count") > 1)

duplicate_count = duplicate_timestamps.count()

print("Duplicate timestamps:", duplicate_count)

if duplicate_count > 0:
    duplicate_timestamps.orderBy(col("timestamp")).show(20, truncate=False)


# ============================================
# 4. REMOVE EXACT DUPLICATE RECORDS
# ============================================

print("\n========== DUPLICATE RECORD REMOVAL ==========")

before_dedup = df.count()

df = df.dropDuplicates()

after_dedup = df.count()

print("Rows before deduplication:", before_dedup)
print("Rows after deduplication:", after_dedup)
print("Rows removed:", before_dedup - after_dedup)


# ============================================
# 5. REMOVE NULL TIMESTAMPS
# ============================================

df = df.filter(col("timestamp").isNotNull())

print("Rows after removing null timestamps:", df.count())


# ============================================
# 6. CHECK TIME RANGE
# ============================================

print("\n========== TIME RANGE ==========")

df.select(
    min("timestamp").alias("start_time"),
    max("timestamp").alias("end_time"),
).show(truncate=False)


# ============================================
# 7. CREATE TIME ORDER INFORMATION
# ============================================

print("\n========== CREATING TIME ORDER ==========")

time_window = Window.orderBy("timestamp")

df = df.withColumn("previous_timestamp", lag("timestamp").over(time_window))

df = df.withColumn(
    "time_since_previous_seconds",
    unix_timestamp("timestamp") - unix_timestamp("previous_timestamp"),
)


# ============================================
# 8. TIME GAP SUMMARY
# ============================================

print("\n========== TIME GAP SUMMARY ==========")

df.select(
    min("time_since_previous_seconds").alias("min_gap_seconds"),
    max("time_since_previous_seconds").alias("max_gap_seconds"),
).show()


# ============================================
# 9. REMOVE TEMPORARY COLUMN
# ============================================

df = df.drop("previous_timestamp")


# ============================================
# 10. SORT CHRONOLOGICALLY
# ============================================

print("\n========== CHRONOLOGICAL SORT ==========")

df = df.orderBy("timestamp")


# ============================================
# 11. VERIFY ORDER
# ============================================

print("\n========== FIRST RECORDS ==========")

df.select(
    "timestamp",
    "_c0",
).show(10, truncate=False)

print("\n========== LAST RECORDS ==========")

df.orderBy(col("timestamp").desc()).select(
    "timestamp",
    "_c0",
).show(10, truncate=False)


# ============================================
# 12. SAVE TIME-SERIES DATASET
# ============================================

print("\n========== SAVING TIME-SERIES DATA ==========")

(df.write.mode("overwrite").parquet(output_path))

print("Output path:", output_path)


print("\n========== TIME-SERIES PREPARATION COMPLETE ==========")

spark.stop()
