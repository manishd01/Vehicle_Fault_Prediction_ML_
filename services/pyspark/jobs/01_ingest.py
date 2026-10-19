from pyspark.sql import SparkSession
import os

import pandas as pd
from pathlib import Path

spark = SparkSession.builder.appName("VehicleFaultDataIngestion").getOrCreate()

# input_path = "/app/data/raw/MetroPT3(AirCompressor).csv"
# output_path = "/app/data/processed/metropt"
# //better way
input_path = os.environ["CSV_PATH"]
output_path = os.environ["PARQUET_PATH"]

# NEW: Diagnose the actual input file
print("CSV_PATH:", repr(input_path))

input_file = Path(input_path)

if not input_file.is_file():
    raise FileNotFoundError(f"Input file does not exist: {input_file}")

with open(input_file, "rb") as f:
    file_signature = f.read(16)

print("First 16 bytes:", file_signature)

# NEW: Detect the actual file format.
# XLSX files are ZIP containers and usually start with PK\x03\x04.
# Do not rely on the filename extension alone.
if file_signature.startswith(b"PK\x03\x04"):
    file_format = "excel"
elif file_signature.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
    # NEW: Older Excel .xls files use the OLE Compound File signature.
    file_format = "excel"
else:
    file_format = "csv"

print("Detected file format:", file_format)
print("Input file:", str(input_file))
print("File extension:", input_file.suffix.lower())

# NEW: Read CSV or Excel files.
# This also supports an Excel file incorrectly named input.csv.
if file_format == "csv":
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .csv(str(input_file))
    )

elif file_format == "excel":
    # NEW: Read Excel using pandas, then convert to a Spark DataFrame.
    # Requires pandas and an Excel engine such as openpyxl in the container.
    pandas_df = pd.read_excel(input_file)
    df = spark.createDataFrame(pandas_df)

else:
    raise ValueError(f"Unsupported input format for file: {input_file}")

print("========== DATASET INFO ==========")
print("Rows:", df.count())
print("Columns:", len(df.columns))

print("\n========== SCHEMA ==========")
df.printSchema()

print("\n========== SAMPLE ==========")
df.show(5, truncate=False)

print("\n========== NULL COUNTS ==========")
for column in df.columns:
    # NEW: Use a quoted column reference so unusual column names are handled safely.
    from pyspark.sql import functions as F

    count = df.filter(F.col(f"`{column.replace('`', '``')}`").isNull()).count()
    print(f"{column}: {count}")

df.write.mode("overwrite").parquet(output_path)

print(f"\nProcessed dataset written to: {output_path}")

spark.stop()
