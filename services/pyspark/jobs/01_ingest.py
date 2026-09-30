from pyspark.sql import SparkSession
import os

spark = SparkSession.builder.appName("VehicleFaultDataIngestion").getOrCreate()

# input_path = "/app/data/raw/MetroPT3(AirCompressor).csv"
# output_path = "/app/data/processed/metropt"
# //better way
input_path = os.environ["CSV_PATH"]
output_path = os.environ["PARQUET_PATH"]

df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
    .csv(input_path)
)

print("========== DATASET INFO ==========")
print("Rows:", df.count())
print("Columns:", len(df.columns))

print("\n========== SCHEMA ==========")
df.printSchema()

print("\n========== SAMPLE ==========")
df.show(5, truncate=False)

print("\n========== NULL COUNTS ==========")
for column in df.columns:
    count = df.filter(df[column].isNull()).count()
    print(f"{column}: {count}")

df.write.mode("overwrite").parquet(output_path)

print(f"\nProcessed dataset written to: {output_path}")

spark.stop()
