from pyspark.sql import SparkSession
import os

spark = SparkSession.builder.appName("VehicleFaultParquetInspection").getOrCreate()

input_path = os.environ["PARQUET_PATH"]


# ==============================
# READ PROCESSED PARQUET
# ==============================

df = spark.read.parquet(input_path)


# ==============================
# BASIC INFORMATION
# ==============================

print("\n========== DATASET INFO ==========")

print("Rows:", df.count())
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns)


# ==============================
# SCHEMA
# ==============================

print("\n========== SCHEMA ==========")

df.printSchema()


# ==============================
# SAMPLE DATA
# ==============================

print("\n========== SAMPLE DATA ==========")

df.show(10, truncate=False)


# ==============================
# NULL COUNTS
# ==============================

print("\n========== NULL COUNTS ==========")

df.select(
    [
        __import__("pyspark")
        .sql.functions.sum(
            __import__("pyspark").sql.functions.col(column).isNull().cast("int")
        )
        .alias(column)
        for column in df.columns
    ]
).show(truncate=False)


# ==============================
# STOP SPARK
# ==============================

spark.stop()
