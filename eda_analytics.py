from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Initialize Spark
spark = SparkSession.builder \
    .appName("UrbanMobility-EDA") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
os.makedirs("visualizations", exist_ok=True)

print("Reading optimized data directly from Parquet Storage Layer...")
df = spark.read.parquet("storage_layer/traffic_parquet")
df.createOrReplaceTempView("traffic_table")

# ==========================================
# VISUALIZATION 1: Hourly Demand Bar Chart
# ==========================================
print("\nExecuting Query 1: Analyzing Hourly Peak Travel Windows...")
hourly_analytics = spark.sql("""
    SELECT pickup_hour, COUNT(*) as total_trips, AVG(fare_amount) as avg_fare
    FROM traffic_table
    GROUP BY pickup_hour
    ORDER BY pickup_hour
""")

print("Downsampling to Pandas for visual generation...")
pandas_hourly = hourly_analytics.toPandas()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
# Fixed the Seaborn warning by adding hue="pickup_hour" and legend=False
sns.barplot(x="pickup_hour", y="total_trips", hue="pickup_hour", data=pandas_hourly, palette="viridis", legend=False)
plt.title("System-Wide Traffic Demand Profile by Hour of Day", fontsize=14)
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Aggregated Total Trips")
plt.savefig("visualizations/hourly_demand.png")
print("Saved analysis visualization: visualizations/hourly_demand.png")


# ==========================================
# VISUALIZATION 2: Weekly Spatiotemporal Heatmap
# ==========================================
print("\nExecuting Query 2: Generating Weekly Spatiotemporal Heatmap Matrix...")
heatmap_query = spark.sql("""
    SELECT day_of_week, pickup_hour, COUNT(*) as total_trips
    FROM traffic_table
    GROUP BY day_of_week, pickup_hour
""")

heatmap_pd = heatmap_query.toPandas()
heatmap_matrix = heatmap_pd.pivot(index='day_of_week', columns='pickup_hour', values='total_trips')

day_map = {1: 'Sun', 2: 'Mon', 3: 'Tue', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat'}
heatmap_matrix.index = heatmap_matrix.index.map(day_map)

plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_matrix, cmap='YlOrRd', linewidths=.5, annot=False)
plt.title('System-Wide Traffic Stress Heatmap (Day vs. Hour)', fontsize=16)
plt.xlabel('Hour of the Day')
plt.ylabel('Day of the Week')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("visualizations/weekly_heatmap.png")
print("Saved analysis visualization: visualizations/weekly_heatmap.png")


# ==========================================
# VISUALIZATION 3: Feature Correlation Matrix
# ==========================================
print("\nExecuting Query 3: Extracting Sample for Correlation Matrix...")
sample_df = df.sample(fraction=0.1, seed=42).select(
    "trip_distance", "fare_amount", "tip_amount", "passenger_count", "pickup_hour"
).toPandas()

plt.figure(figsize=(10, 8))
corr_matrix = sample_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
plt.title('Feature Correlation Matrix (10% Distributed Sample)', fontsize=14)
plt.tight_layout()
plt.savefig("visualizations/correlation_matrix.png")
print("Saved analysis visualization: visualizations/correlation_matrix.png")


# Shut down the Spark engine ONLY after all queries are finished
spark.stop()
print("\nAll Exploratory Data Analysis complete!")
