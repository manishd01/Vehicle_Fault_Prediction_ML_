from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, sum as spark_sum
import os

# ==============================
# SPARK SESSION
# ==============================

spark = SparkSession.builder.appName("VehicleFaultTargetCreation").getOrCreate()


# ==============================
# PATH CONFIGURATION
# ==============================

data_root = os.environ["DATA_ROOT"]

input_path = f"{data_root}/processed/metropt_timeseries"
output_path = f"{data_root}/processed/metropt_labeled"


# ==============================
# READ INPUT DATA
# ==============================

print("\n========== READING TIME SERIES DATA ==========")
print("Input:", input_path)

df = spark.read.parquet(input_path)

print("Rows:", df.count())


# ==============================
# DOCUMENTED FAILURE WINDOWS
# ==============================

failure_windows = [
    ("F1", "2020-04-18 00:00:00", "2020-04-18 23:59:59"),
    ("F2", "2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("F3", "2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("F4", "2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]


# ==============================
# CREATE TARGET LABEL
# ==============================

print("\n========== CREATING TARGET LABEL ==========")

fault_condition = None

for failure_id, start_time, end_time in failure_windows:

    condition = (col("timestamp") >= lit(start_time).cast("timestamp")) & (
        col("timestamp") <= lit(end_time).cast("timestamp")
    )

    if fault_condition is None:
        fault_condition = condition
    else:
        fault_condition = fault_condition | condition


df = df.withColumn("fault_label", when(fault_condition, lit(1)).otherwise(lit(0)))


# ==============================
# CREATE FAILURE ID
# ==============================

failure_id_expression = lit(0)

for failure_id, start_time, end_time in failure_windows:

    condition = (col("timestamp") >= lit(start_time).cast("timestamp")) & (
        col("timestamp") <= lit(end_time).cast("timestamp")
    )

    failure_id_expression = when(condition, lit(failure_id)).otherwise(
        failure_id_expression
    )


df = df.withColumn("failure_id", failure_id_expression)


# ==============================
# TARGET DISTRIBUTION
# ==============================

print("\n========== TARGET DISTRIBUTION ==========")

df.groupBy("fault_label").count().orderBy("fault_label").show()


# ==============================
# FAILURE DISTRIBUTION
# ==============================

print("\n========== FAILURE DISTRIBUTION ==========")

df.groupBy("failure_id").count().orderBy("failure_id").show()


# ==============================
# FAILURE PERIOD SUMMARY
# ==============================

print("\n========== FAILURE PERIOD SUMMARY ==========")

df.filter(col("fault_label") == 1).select("timestamp", "failure_id").orderBy(
    "timestamp"
).show(20, truncate=False)


# ==============================
# TARGET PERCENTAGE
# ==============================

print("\n========== TARGET PERCENTAGE ==========")

total_rows = df.count()

fault_rows = df.filter(col("fault_label") == 1).count()

fault_percentage = fault_rows / total_rows * 100 if total_rows > 0 else 0

print(f"Total rows: {total_rows}")
print(f"Fault rows: {fault_rows}")
print(f"Fault percentage: {fault_percentage:.4f}%")


# ==============================
# WRITE OUTPUT
# ==============================

print("\n========== WRITING LABELED DATA ==========")
print("Output:", output_path)

df.write.mode("overwrite").parquet(output_path)

print("\nTarget creation completed successfully.")


# ==============================
# STOP SPARK
# ==============================

spark.stop()
