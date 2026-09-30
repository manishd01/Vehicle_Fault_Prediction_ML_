# 08_feature_engineering.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lag,
    avg,
    stddev,
    min as spark_min,
    max as spark_max,
    hour,
    minute,
    dayofweek,
    sin,
    cos,
)
from pyspark.sql.window import Window
import os

# ==============================
# SPARK SESSION
# ==============================

spark = SparkSession.builder.appName("VehicleFaultFeatureEngineering").getOrCreate()


# ==============================
# PATH CONFIGURATION
# ==============================

data_root = os.environ["DATA_ROOT"]

input_path = f"{data_root}/processed/metropt_labeled"
output_path = f"{data_root}/processed/metropt_features"


# ==============================
# READ LABELED DATA
# ==============================

print("\n========== READING LABELED DATA ==========")
print("Input:", input_path)

df = spark.read.parquet(input_path)

print("Rows:", df.count())


# ==============================
# SENSOR COLUMNS
# ==============================

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


# ==============================
# WINDOW DEFINITIONS
# ==============================

time_window = Window.orderBy("timestamp")

rolling_window = Window.orderBy("timestamp").rowsBetween(-29, 0)


# ==============================
# LAG FEATURES
# ==============================

print("\n========== CREATING LAG FEATURES ==========")

for sensor in sensor_columns:

    previous_column = f"{sensor}_previous"

    df = df.withColumn(previous_column, lag(col(sensor), 1).over(time_window))

    df = df.withColumn(f"{sensor}_delta", col(sensor) - col(previous_column))


# ==============================
# ROLLING FEATURES
# ==============================

print("\n========== CREATING ROLLING FEATURES ==========")

for sensor in sensor_columns:

    df = df.withColumn(
        f"{sensor}_rolling_mean_5m", avg(col(sensor)).over(rolling_window)
    )

    df = df.withColumn(
        f"{sensor}_rolling_std_5m", stddev(col(sensor)).over(rolling_window)
    )

    df = df.withColumn(
        f"{sensor}_rolling_min_5m", spark_min(col(sensor)).over(rolling_window)
    )

    df = df.withColumn(
        f"{sensor}_rolling_max_5m", spark_max(col(sensor)).over(rolling_window)
    )


# ==============================
# TIME FEATURES
# ==============================

print("\n========== CREATING TIME FEATURES ==========")

df = df.withColumn("hour", hour(col("timestamp")))

df = df.withColumn("minute", minute(col("timestamp")))

df = df.withColumn("day_of_week", dayofweek(col("timestamp")))


# ==============================
# CYCLICAL TIME FEATURES
# ==============================

df = df.withColumn("hour_sin", sin(2 * 3.141592653589793 * col("hour") / 24))

df = df.withColumn("hour_cos", cos(2 * 3.141592653589793 * col("hour") / 24))


# ==============================
# REMOVE TEMPORARY LAG COLUMNS
# ==============================

for sensor in sensor_columns:

    df = df.drop(f"{sensor}_previous")


# ==============================
# SHOW FEATURE DATA
# ==============================

print("\n========== FEATURE DATA ==========")

print("Total columns:", len(df.columns))

print("\nColumns:")
print(df.columns)

print("\nSample:")
df.show(10, truncate=False)


# ==============================
# WRITE FEATURES
# ==============================

print("\n========== WRITING FEATURE DATA ==========")
print("Output:", output_path)

df.write.mode("overwrite").parquet(output_path)

print("\nFeature engineering completed successfully.")


# ==============================
# STOP SPARK
# ==============================

spark.stop()
