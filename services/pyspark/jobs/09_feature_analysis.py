from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, sum as spark_sum
import os

# ==============================
# SPARK SESSION
# ==============================

spark = SparkSession.builder.appName("VehicleFaultFeatureAnalysis").getOrCreate()


# ==============================
# PATH CONFIGURATION
# ==============================

data_root = os.environ["DATA_ROOT"]

input_path = f"{data_root}/processed/metropt_features"


# ==============================
# READ FEATURE DATA
# ==============================

print("\n========== READING FEATURE DATA ==========")
print("Input:", input_path)

df = spark.read.parquet(input_path)

print("Rows:", df.count())
print("Columns:", len(df.columns))


# ==============================
# SCHEMA
# ==============================

print("\n========== SCHEMA ==========")

df.printSchema()


# ==============================
# TARGET DISTRIBUTION
# ==============================

print("\n========== TARGET DISTRIBUTION ==========")

df.groupBy("fault_label").count().orderBy("fault_label").show()


# ==============================
# NUMERIC FEATURES
# ==============================

numeric_types = {"int", "bigint", "double", "float", "long", "smallint", "tinyint"}

numeric_columns = [
    field.name
    for field in df.schema.fields
    if field.dataType.simpleString() in numeric_types
]

print("\n========== NUMERIC FEATURES ==========")

print("Number of numeric columns:", len(numeric_columns))

print(numeric_columns)


# ==============================
# DESCRIPTIVE STATISTICS
# ==============================

print("\n========== DESCRIPTIVE STATISTICS ==========")

df.select(numeric_columns).describe().show(truncate=False)


# ==============================
# FEATURE MEANS BY TARGET
# ==============================

important_features = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "Caudal_impulses",
]

available_features = [
    feature for feature in important_features if feature in df.columns
]

print("\n========== FEATURE MEANS BY TARGET ==========")

df.groupBy("fault_label").agg(
    *[avg(col(feature)).alias(f"{feature}_mean") for feature in available_features]
).orderBy("fault_label").show(truncate=False)


# ==============================
# CORRELATION WITH TARGET
# ==============================

print("\n========== FEATURE CORRELATION WITH TARGET ==========")

for feature in numeric_columns:

    if feature in ["fault_label", "failure_id"]:
        continue

    try:
        correlation = df.stat.corr(feature, "fault_label")

        print(f"{feature}: " f"{correlation}")

    except Exception as error:

        print(f"{feature}: " f"Correlation unavailable - {error}")


# ==============================
# OPERATING STATE ANALYSIS
# ==============================

operating_state_columns = ["TP2", "TP3", "H1", "DV_pressure"]

for feature in operating_state_columns:

    if feature not in df.columns:
        continue

    print(f"\n========== {feature} BY TARGET ==========")

    df.groupBy("fault_label").agg(
        avg(col(feature)).alias("mean"),
        spark_sum(col(feature).isNull().cast("int")).alias("null_count"),
        count(col(feature)).alias("non_null_count"),
    ).orderBy("fault_label").show()


# ==============================
# MISSING VALUES
# ==============================

print("\n========== MISSING VALUES ==========")

null_expressions = [
    spark_sum(col(column).isNull().cast("int")).alias(column) for column in df.columns
]

df.select(null_expressions).show(truncate=False)


# ==============================
# CONSTANT FEATURES
# ==============================

print("\n========== CONSTANT FEATURES ==========")

for feature in numeric_columns:

    if feature in ["fault_label", "failure_id"]:
        continue

    distinct_count = df.select(feature).distinct().count()

    if distinct_count <= 1:

        print(f"Constant feature: {feature}")


print("\nFeature analysis completed successfully.")


# ==============================
# STOP SPARK
# ==============================

spark.stop()
