from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max
import os

spark = SparkSession.builder.appName("VehicleFaultDataValidation").getOrCreate()

input_path = os.environ["PARQUET_PATH"]

df = spark.read.parquet(input_path)


# ============================================
# 1. DATASET SIZE
# ============================================

print("\n========== DATASET SIZE ==========")

print("Rows:", df.count())
print("Columns:", len(df.columns))


# ============================================
# 2. DATA TYPES
# ============================================

print("\n========== DATA TYPES ==========")

df.printSchema()


# ============================================
# 3. TIMESTAMP RANGE
# ============================================

print("\n========== TIMESTAMP RANGE ==========")

df.select(
    min("timestamp").alias("min_timestamp"), max("timestamp").alias("max_timestamp")
).show(truncate=False)


# ============================================
# 4. INDEX COLUMN CHECK
# ============================================

print("\n========== INDEX COLUMN CHECK ==========")

df.select(min("_c0").alias("min_index"), max("_c0").alias("max_index")).show()

print("Total rows:", df.count())


# ============================================
# 5. NUMERIC SENSOR STATISTICS
# ============================================

print("\n========== SENSOR STATISTICS ==========")

numeric_columns = [
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

df.select(
    [min(col(c)).alias(f"{c}_min") for c in numeric_columns]
    + [max(col(c)).alias(f"{c}_max") for c in numeric_columns]
).show(truncate=False)


spark.stop()


# //checking
