from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    row_number,
    count,
    rand,
    floor,
    greatest,
    least,
    lit,
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

print("\n========== STRATIFIED DATA SPLIT ==========")

total_rows = df.count()
print("Total rows:", total_rows)

# Each fault_label class must have at least 3 rows
class_counts = df.groupBy("fault_label").count()
class_counts.show()

too_small = class_counts.filter(col("count") < 3).count()

if too_small > 0:
    raise ValueError("Stratified split requires at least 3 rows per fault_label class.")

# Randomly order rows separately within each class
class_window = Window.partitionBy("fault_label")
random_window = Window.partitionBy("fault_label").orderBy(rand(42))

df = df.withColumn("_split_row", row_number().over(random_window))
df = df.withColumn("_class_total", count("*").over(class_window))

# Allocate approximately 70% to train, 15% to validation, 15% to test
df = df.withColumn(
    "_train_end",
    greatest(lit(1), floor(col("_class_total") * 0.70)).cast("int"),
)

df = df.withColumn(
    "_train_end",
    least(col("_train_end"), col("_class_total") - 2),
)

df = df.withColumn(
    "_validation_end",
    greatest(
        col("_train_end") + 1,
        floor(col("_class_total") * 0.85),
    ).cast("int"),
)

df = df.withColumn(
    "_validation_end",
    least(col("_validation_end"), col("_class_total") - 1),
)

train_df = df.filter(col("_split_row") <= col("_train_end")).drop(
    "_split_row", "_class_total", "_train_end", "_validation_end"
)

validation_df = df.filter(
    (col("_split_row") > col("_train_end"))
    & (col("_split_row") <= col("_validation_end"))
).drop("_split_row", "_class_total", "_train_end", "_validation_end")

test_df = df.filter(col("_split_row") > col("_validation_end")).drop(
    "_split_row", "_class_total", "_train_end", "_validation_end"
)

print("\n========== SPLIT BOUNDARIES ==========")
print("Train: approximately 70%")
print("Validation: approximately 15%")
print("Test: approximately 15%")

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

# print("\n========== TEMPORAL LEAKAGE CHECK ==========")

# train_max = train_df.select(spark_max("timestamp")).collect()[0][0]

# validation_min = validation_df.select(spark_min("timestamp")).collect()[0][0]

# validation_max = validation_df.select(spark_max("timestamp")).collect()[0][0]

# test_min = test_df.select(spark_min("timestamp")).collect()[0][0]


# print("Train max:", train_max)
# print("Validation min:", validation_min)
# print("Validation max:", validation_max)
# print("Test min:", test_min)


# if train_max <= validation_min:

#     print("Train → Validation temporal ordering: PASS")

# else:

#     print("Train → Validation temporal ordering: FAIL")


# if validation_max <= test_min:

#     print("Validation → Test temporal ordering: PASS")

# else:

#     print("Validation → Test temporal ordering: FAIL")


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
