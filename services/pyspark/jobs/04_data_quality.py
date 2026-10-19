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

# ============================================
# INDEX CHECK
# ============================================

print("\n========== INDEX CHECK ==========")

if "Unnamed: 0" in df.columns:
    index_col = "Unnamed: 0"
elif "_c0" in df.columns:
    index_col = "_c0"
else:
    index_col = None

if index_col is not None:
    index_stats = df.selectExpr(
        f"COUNT(`{index_col}`) AS total_rows",
        f"COUNT(DISTINCT `{index_col}`) AS distinct_indexes",
        f"MIN(`{index_col}`) AS min_index",
        f"MAX(`{index_col}`) AS max_index",
    )

    index_stats.show(truncate=False)
    print(f"Index column checked: {index_col}")

else:
    print("WARNING: No recognised index column found.")
    print(f"Available columns: {df.columns}")


# ============================================
# 4. CHECK DUPLICATE INDEX VALUES
# ============================================

print("\n========== DUPLICATE INDEX CHECK ==========")

if index_col:
    duplicate_index = df.groupBy(index_col).count().filter(col("count") > 1).limit(10)
    duplicate_index.show(truncate=False)
else:
    print("WARNING: No recognised index column found.")

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
