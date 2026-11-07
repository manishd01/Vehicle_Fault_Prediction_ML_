# Databricks notebook source

from pyspark.sql import functions as F

# ============================================================
# 1. TABLE CONFIGURATION
# ============================================================

INPUT_TABLE = "vehicle_fault_catalog.vehicle_fault_prediction.gold_features"

TRAIN_TABLE = "vehicle_fault_catalog.vehicle_fault_prediction.gold_train"

VALIDATION_TABLE = "vehicle_fault_catalog.vehicle_fault_prediction.gold_validation"

TEST_TABLE = "vehicle_fault_catalog.vehicle_fault_prediction.gold_test"


# ============================================================
# 2. READ FEATURE DATA
# ============================================================

print("\n========== READING GOLD FEATURES ==========")

df = spark.table(INPUT_TABLE)

total_rows = df.count()

print("Input table:", INPUT_TABLE)
print("Total rows:", total_rows)
print("Total columns:", len(df.columns))

if total_rows == 0:
    raise ValueError("Feature dataset is empty.")

required_columns = ["timestamp", "fault_label"]

missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")


# ============================================================
# 3. VALIDATE TIMESTAMPS AND LABELS
# ============================================================

print("\n========== DATA VALIDATION ==========")

null_timestamps = df.filter(F.col("timestamp").isNull()).count()

null_labels = df.filter(F.col("fault_label").isNull()).count()

invalid_labels = df.filter(~F.col("fault_label").isin(0, 1)).count()

print("Null timestamps:", null_timestamps)
print("Null labels:", null_labels)
print("Invalid labels:", invalid_labels)

if null_timestamps > 0 or null_labels > 0 or invalid_labels > 0:
    raise ValueError("Dataset contains invalid timestamps or target labels.")


# ============================================================
# 4. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n========== OVERALL TARGET DISTRIBUTION ==========")

df.groupBy("fault_label").count().orderBy("fault_label").show()


# ============================================================
# 5. CALCULATE CHRONOLOGICAL SPLIT BOUNDARIES
# ============================================================

print("\n========== CALCULATING SPLIT BOUNDARIES ==========")

# Use approximate timestamp percentiles instead of a global
# row_number window. This avoids requiring a full global sort
# just to calculate the split boundaries.

split_points = df.selectExpr(
    """
    percentile_approx(
        CAST(timestamp AS BIGINT),
        0.70,
        10000
    ) AS train_boundary
    """,
    """
    percentile_approx(
        CAST(timestamp AS BIGINT),
        0.85,
        10000
    ) AS validation_boundary
    """,
).first()

train_boundary = split_points["train_boundary"]
validation_boundary = split_points["validation_boundary"]

if train_boundary is None or validation_boundary is None:
    raise ValueError("Unable to calculate chronological split boundaries.")

if train_boundary >= validation_boundary:
    raise ValueError("Invalid chronological split boundaries.")

print("Training boundary (Unix seconds):", train_boundary)
print("Validation boundary (Unix seconds):", validation_boundary)


# ============================================================
# 6. CREATE CHRONOLOGICAL DATA SPLITS
# ============================================================

print("\n========== CREATING DATA SPLITS ==========")

timestamp_seconds = F.col("timestamp").cast("long")

train_df = df.filter(timestamp_seconds <= F.lit(train_boundary))

validation_df = df.filter(
    (timestamp_seconds > F.lit(train_boundary))
    & (timestamp_seconds <= F.lit(validation_boundary))
)

test_df = df.filter(timestamp_seconds > F.lit(validation_boundary))


# ============================================================
# 7. VERIFY SPLIT COUNTS
# ============================================================

print("\n========== SPLIT COUNTS ==========")

train_count = train_df.count()
validation_count = validation_df.count()
test_count = test_df.count()

print("Total rows:", total_rows)
print("Train rows:", train_count)
print("Validation rows:", validation_count)
print("Test rows:", test_count)

if train_count == 0 or validation_count == 0 or test_count == 0:
    raise ValueError("At least one chronological split is empty.")

if train_count + validation_count + test_count != total_rows:
    raise ValueError("Split row counts do not match the original dataset.")


# ============================================================
# 8. TARGET DISTRIBUTION PER SPLIT
# ============================================================

print("\n========== TRAIN TARGET DISTRIBUTION ==========")

train_df.groupBy("fault_label").count().orderBy("fault_label").show()


print("\n========== VALIDATION TARGET DISTRIBUTION ==========")

validation_df.groupBy("fault_label").count().orderBy("fault_label").show()


print("\n========== TEST TARGET DISTRIBUTION ==========")

test_df.groupBy("fault_label").count().orderBy("fault_label").show()


# ============================================================
# 9. CHECK FAULT CLASS AVAILABILITY
# ============================================================

print("\n========== CLASS AVAILABILITY CHECK ==========")


def check_fault_classes(dataset, dataset_name):

    present_classes = {
        row["fault_label"] for row in dataset.select("fault_label").distinct().collect()
    }

    if 0 not in present_classes or 1 not in present_classes:
        print(
            f"WARNING: {dataset_name} does not contain both "
            "fault classes. Classification metrics may be unreliable."
        )
    else:
        print(f"{dataset_name}: Both normal and fault classes are present.")


check_fault_classes(train_df, "Training")
check_fault_classes(validation_df, "Validation")
check_fault_classes(test_df, "Testing")


# ============================================================
# 10. DATE RANGE VERIFICATION
# ============================================================

print("\n========== TRAIN DATE RANGE ==========")

train_df.agg(
    F.min("timestamp").alias("start_timestamp"),
    F.max("timestamp").alias("end_timestamp"),
).show(truncate=False)


print("\n========== VALIDATION DATE RANGE ==========")

validation_df.agg(
    F.min("timestamp").alias("start_timestamp"),
    F.max("timestamp").alias("end_timestamp"),
).show(truncate=False)


print("\n========== TEST DATE RANGE ==========")

test_df.agg(
    F.min("timestamp").alias("start_timestamp"),
    F.max("timestamp").alias("end_timestamp"),
).show(truncate=False)


# ============================================================
# 11. SAVE TRAIN DATASET
# ============================================================

print("\n========== SAVING TRAIN DATA ==========")

(
    train_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TRAIN_TABLE)
)


# ============================================================
# 12. SAVE VALIDATION DATASET
# ============================================================

print("\n========== SAVING VALIDATION DATA ==========")

(
    validation_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(VALIDATION_TABLE)
)


# ============================================================
# 13. SAVE TEST DATASET
# ============================================================

print("\n========== SAVING TEST DATA ==========")

(
    test_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TEST_TABLE)
)


# ============================================================
# 14. VERIFY SAVED TABLES
# ============================================================

print("\n========== VERIFYING SAVED TABLES ==========")

for table_name in [TRAIN_TABLE, VALIDATION_TABLE, TEST_TABLE]:

    saved_df = spark.table(table_name)

    print(table_name, "rows:", saved_df.count())


print("\n========== GOLD DATASET PREPARATION COMPLETE ==========")
