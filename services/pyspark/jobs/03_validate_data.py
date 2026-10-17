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


# ============================================================
# INDEX COLUMN CHECK
# ============================================================

print("\n========== INDEX COLUMN CHECK ==========")

# The CSV index was saved as "Unnamed: 0" in the Parquet dataset.
# Older code expected "_c0", which does not exist here.

if "Unnamed: 0" in df.columns:
    index_col = "Unnamed: 0"

    df.select(
        min(index_col).alias("min_index"), max(index_col).alias("max_index")
    ).show()

    print(f"Index column found: {index_col}")

elif "_c0" in df.columns:
    # Keep compatibility with datasets that use the older column name.
    index_col = "_c0"

    df.select(
        min(index_col).alias("min_index"), max(index_col).alias("max_index")
    ).show()

    print(f"Index column found: {index_col}")

else:
    print("WARNING: No recognised index column found.")
    print(f"Available columns: {df.columns}")


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
