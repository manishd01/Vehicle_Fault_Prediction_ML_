# NEW FILE:
# services/pyspark/jobs/04_data_quality.py

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    min,
    max,
)

spark = SparkSession.builder.appName("VehicleFaultDataQuality").getOrCreate()

input_path = os.environ["PARQUET_PATH"]

df = spark.read.parquet(input_path)
# hi test


# ============================================
# 1. NULL CHECK
# ============================================

print("\n========== NULL CHECK ==========")

null_counts = df.select([count(col(c)).alias(c) for c in []])

for column in df.columns:
    null_count = df.filter(col(column).isNull()).count()
    print(f"{column}: {null_count}")


# ============================================
# 2. TIMESTAMP NULL CHECK
# ============================================

print("\n========== TIMESTAMP CHECK ==========")

timestamp_stats = df.select(
    count("*").alias("total_rows"),
    count("timestamp").alias("valid_timestamps"),
    min("timestamp").alias("min_timestamp"),
    max("timestamp").alias("max_timestamp"),
).first()

print("Total rows:", timestamp_stats["total_rows"])
print("Valid timestamps:", timestamp_stats["valid_timestamps"])
print(
    "Null timestamps:",
    timestamp_stats["total_rows"] - timestamp_stats["valid_timestamps"],
)
print("Min timestamp:", timestamp_stats["min_timestamp"])
print("Max timestamp:", timestamp_stats["max_timestamp"])


# ============================================
# 3. INDEX UNIQUENESS CHECK
# ============================================

print("\n========== INDEX CHECK ==========")

index_stats = df.select(
    count("*").alias("total_rows"),
    countDistinct("_c0").alias("distinct_indexes"),
    min("_c0").alias("min_index"),
    max("_c0").alias("max_index"),
).first()

print("Total rows:", index_stats["total_rows"])
print("Distinct _c0:", index_stats["distinct_indexes"])
print("Min _c0:", index_stats["min_index"])
print("Max _c0:", index_stats["max_index"])


# ============================================
# 4. CHECK DUPLICATE INDEX VALUES
# ============================================

print("\n========== DUPLICATE INDEX CHECK ==========")

duplicate_index = df.groupBy("_c0").count().filter(col("count") > 1).limit(10)

duplicate_index.show(truncate=False)


# ============================================
# 5. TIMESTAMP DUPLICATES
# ============================================

print("\n========== DUPLICATE TIMESTAMP CHECK ==========")

duplicate_timestamps = (
    df.groupBy("timestamp").count().filter(col("count") > 1).limit(10)
)

duplicate_timestamps.show(truncate=False)


# ============================================
# 6. BINARY / FLAG COLUMN CHECK
# ============================================

print("\n========== FLAG COLUMN RANGES ==========")

flag_columns = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
]

df.select(
    [min(col(c)).alias(f"{c}_min") for c in flag_columns]
    + [max(col(c)).alias(f"{c}_max") for c in flag_columns]
).show(truncate=False)


# ============================================
# 7. SENSOR NULL SUMMARY
# ============================================

print("\n========== SENSOR NULL SUMMARY ==========")

sensor_columns = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

null_summary = df.select([count(col(c)).alias(c) for c in sensor_columns]).first()

for column in sensor_columns:
    valid_count = null_summary[column]
    null_count = index_stats["total_rows"] - valid_count
    print(f"{column}: {null_count}")


print("\n========== DATA QUALITY CHECK COMPLETE ==========")

spark.stop()
