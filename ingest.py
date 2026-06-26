from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, when
import time

# Initialize Distributed Spark Session
spark = SparkSession.builder \
    .appName("UrbanMobility-Ingestion") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
start_time = time.time()

print("Step 1: Reading Raw CSV via Spark Engine...")
raw_df = spark.read.csv("raw_traffic_data.csv", header=True, inferSchema=True)

print(f"Total Raw Record Count: {raw_df.count()}")

print("\nStep 2: Performing Big Data Pre-processing & Filtering...")
# Data cleaning: Drop records violating real-world validation boundaries
cleaned_df = raw_df.filter(
    (col("trip_distance") > 0.1) & (col("trip_distance") < 100.0) &
    (col("fare_amount") >= 2.5) & (col("fare_amount") < 500.0) &
    (col("passenger_count") > 0)
)

print("\nStep 3: Engineering Temporal & Categorical Features...")
processed_df = cleaned_df \
    .withColumn("pickup_hour", hour(col("pickup_datetime"))) \
    .withColumn("day_of_week", dayofweek(col("pickup_datetime"))) \
    .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0))

print("\nStep 4: Writing Cleaned Records to Partitioned Columnar Parquet Store...")
# Partitioning by day_of_week minimizes block scanning during localized queries
processed_df.write \
    .mode("overwrite") \
    .partitionBy("day_of_week") \
    .parquet("storage_layer/traffic_parquet")

print(f"\nIngestion and Pre-processing pipeline executed successfully in {time.time() - start_time:.2f} seconds!")
spark.stop()
