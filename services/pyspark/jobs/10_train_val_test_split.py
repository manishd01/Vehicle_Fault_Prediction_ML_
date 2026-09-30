from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    row_number,
    count,
    min as spark_min,
    max as spark_max,
)
from pyspark.sql.window import Window
import os

# ==============================
# SPARK SESSION
# ==============================

spark = SparkSession.builder.appName(
    "VehicleFaultTrainValidationTestSplit"
).getOrCreate()


# ==============================
# PATH CONFIGURATION
# ==============================

data_root = os.environ["DATA_ROOT"]

input_path = f"{data_root}/processed/metropt_features"

train_path = f"{data_root}/processed/train"
validation_path = f"{data_root}/processed/validation"
test_path = f"{data_root}/processed/test"


# ==============================
# READ FEATURE DATA
# ==============================

print("\n========== READING FEATURE DATA ==========")
print("Input:", input_path)

df = spark.read.parquet(input_path)

print("Total rows:", df.count())


# ==============================
# ORDER DATA CHRONOLOGICALLY
# ==============================

print("\n========== ORDERING DATA ==========")

window = Window.orderBy("timestamp")

df = df.withColumn("row_number", row_number().over(window))


# ==============================
# TOTAL ROW COUNT
# ==============================

total_rows = df.count()

print("Total rows:", total_rows)


# ==============================
# SPLIT BOUNDARIES
# ==============================

train_end = int(total_rows * 0.70)

validation_end = int(total_rows * 0.85)

print("\n========== SPLIT BOUNDARIES ==========")

print(f"Train: 1 - {train_end}")

print(f"Validation: " f"{train_end + 1} - {validation_end}")

print(f"Test: " f"{validation_end + 1} - {total_rows}")


# ==============================
# CREATE SPLITS
# ==============================

train_df = df.filter(col("row_number") <= train_end).drop("row_number")

validation_df = df.filter(
    (col("row_number") > train_end) & (col("row_number") <= validation_end)
).drop("row_number")

test_df = df.filter(col("row_number") > validation_end).drop("row_number")


# ==============================
# SPLIT COUNTS
# ==============================

print("\n========== SPLIT COUNTS ==========")

print("Train rows:", train_df.count())

print("Validation rows:", validation_df.count())

print("Test rows:", test_df.count())


# ==============================
# TARGET DISTRIBUTION
# ==============================

print("\n========== TRAIN TARGET ==========")

train_df.groupBy("fault_label").count().orderBy("fault_label").show()


print("\n========== VALIDATION TARGET ==========")

validation_df.groupBy("fault_label").count().orderBy("fault_label").show()


print("\n========== TEST TARGET ==========")

test_df.groupBy("fault_label").count().orderBy("fault_label").show()


# ==============================
# DATE RANGES
# ==============================

print("\n========== TRAIN DATE RANGE ==========")

train_df.select(
    spark_min("timestamp").alias("min_timestamp"),
    spark_max("timestamp").alias("max_timestamp"),
).show()


print("\n========== VALIDATION DATE RANGE ==========")

validation_df.select(
    spark_min("timestamp").alias("min_timestamp"),
    spark_max("timestamp").alias("max_timestamp"),
).show()


print("\n========== TEST DATE RANGE ==========")

test_df.select(
    spark_min("timestamp").alias("min_timestamp"),
    spark_max("timestamp").alias("max_timestamp"),
).show()


# ==============================
# TEMPORAL LEAKAGE CHECK
# ==============================

print("\n========== TEMPORAL LEAKAGE CHECK ==========")

train_max = train_df.select(spark_max("timestamp")).collect()[0][0]

validation_min = validation_df.select(spark_min("timestamp")).collect()[0][0]

validation_max = validation_df.select(spark_max("timestamp")).collect()[0][0]

test_min = test_df.select(spark_min("timestamp")).collect()[0][0]


print("Train max:", train_max)
print("Validation min:", validation_min)
print("Validation max:", validation_max)
print("Test min:", test_min)


if train_max <= validation_min:

    print("Train → Validation temporal ordering: PASS")

else:

    print("Train → Validation temporal ordering: FAIL")


if validation_max <= test_min:

    print("Validation → Test temporal ordering: PASS")

else:

    print("Validation → Test temporal ordering: FAIL")


# ==============================
# WRITE DATASETS
# ==============================

print("\n========== WRITING TRAIN DATA ==========")
print("Output:", train_path)

train_df.write.mode("overwrite").parquet(train_path)


print("\n========== WRITING VALIDATION DATA ==========")
print("Output:", validation_path)

validation_df.write.mode("overwrite").parquet(validation_path)


print("\n========== WRITING TEST DATA ==========")
print("Output:", test_path)

test_df.write.mode("overwrite").parquet(test_path)


print("\nTrain/validation/test split completed successfully.")


# ==============================
# STOP SPARK
# ==============================

spark.stop()
