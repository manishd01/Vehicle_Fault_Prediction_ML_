# Databricks notebook source

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ============================================================
# 1. TABLE CONFIGURATION
# ============================================================

INPUT_TABLE = "vehicle_fault_catalog.vehicle_fault_prediction.gold_labeled_sensor"

OUTPUT_TABLE = "vehicle_fault_catalog.vehicle_fault_prediction.gold_features"


# ============================================================
# 2. SENSOR COLUMNS
# ============================================================

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


# ============================================================
# 3. READ LABELED DATA
# ============================================================

print("\n========== READING LABELED DATA ==========")

df = spark.table(INPUT_TABLE)

print("Input table:", INPUT_TABLE)
print("Rows:", df.count())
print("Columns:", len(df.columns))


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

print("\n========== VALIDATING SENSOR COLUMNS ==========")

missing_columns = [column for column in sensor_columns if column not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required sensor columns: {missing_columns}")

if "timestamp" not in df.columns:
    raise ValueError("Timestamp column is missing.")

if "fault_label" not in df.columns:
    raise ValueError("Fault label column is missing.")


# ============================================================
# 5. WINDOW DEFINITIONS
# ============================================================

print("\n========== DEFINING WINDOWS ==========")

order_columns = [F.col("timestamp").asc()]

if "_c0" in df.columns:
    order_columns.append(F.col("_c0").asc())

time_window = Window.orderBy(*order_columns)

# Current row + previous 29 rows = 30 observations.
# This is approximately 5 minutes only when sampling is
# consistently every 10 seconds.

rolling_window = Window.orderBy(*order_columns).rowsBetween(-29, 0)


# ============================================================
# 6. LAG FEATURES
# ============================================================

print("\n========== CREATING LAG FEATURES ==========")

for sensor in sensor_columns:

    previous_column = f"{sensor}_previous"

    df = df.withColumn(previous_column, F.lag(F.col(sensor), 1).over(time_window))

    df = df.withColumn(f"{sensor}_delta", F.col(sensor) - F.col(previous_column))


# ============================================================
# 7. ROLLING FEATURES
# ============================================================

print("\n========== CREATING ROLLING FEATURES ==========")

for sensor in sensor_columns:

    df = df.withColumn(
        f"{sensor}_rolling_mean_5m", F.avg(F.col(sensor)).over(rolling_window)
    )

    df = df.withColumn(
        f"{sensor}_rolling_std_5m", F.stddev(F.col(sensor)).over(rolling_window)
    )

    df = df.withColumn(
        f"{sensor}_rolling_min_5m", F.min(F.col(sensor)).over(rolling_window)
    )

    df = df.withColumn(
        f"{sensor}_rolling_max_5m", F.max(F.col(sensor)).over(rolling_window)
    )


# ============================================================
# 8. TIME FEATURES
# ============================================================

print("\n========== CREATING TIME FEATURES ==========")

df = df.withColumn("hour", F.hour("timestamp"))

df = df.withColumn("minute", F.minute("timestamp"))

df = df.withColumn("day_of_week", F.dayofweek("timestamp"))


# ============================================================
# 9. CYCLICAL TIME FEATURES
# ============================================================

print("\n========== CREATING CYCLICAL FEATURES ==========")

pi = 3.141592653589793

df = df.withColumn("hour_sin", F.sin(2 * pi * F.col("hour") / 24))

df = df.withColumn("hour_cos", F.cos(2 * pi * F.col("hour") / 24))


# ============================================================
# 10. REMOVE TEMPORARY LAG COLUMNS
# ============================================================

print("\n========== REMOVING TEMPORARY COLUMNS ==========")

for sensor in sensor_columns:
    df = df.drop(f"{sensor}_previous")


# ============================================================
# 11. FEATURE VALIDATION
# ============================================================

print("\n========== FEATURE VALIDATION ==========")

print("Total rows:", df.count())
print("Total columns:", len(df.columns))

print("\nFeature columns:")

generated_features = [
    column
    for column in df.columns
    if (
        "_delta" in column
        or "_rolling_" in column
        or column in ["hour", "minute", "day_of_week", "hour_sin", "hour_cos"]
    )
]

print(generated_features)

print("\nSample feature data:")

df.select("timestamp", "fault_label", *generated_features[:10]).show(10, truncate=False)


# ============================================================
# 12. SAVE GOLD FEATURES TABLE
# ============================================================

print("\n========== SAVING GOLD FEATURES ==========")

(
    df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE)
)


# ============================================================
# 13. VERIFY OUTPUT
# ============================================================

saved_df = spark.table(OUTPUT_TABLE)

print("Output table:", OUTPUT_TABLE)
print("Saved rows:", saved_df.count())
print("Saved columns:", len(saved_df.columns))

print("\n========== FEATURE ENGINEERING COMPLETE ==========")
