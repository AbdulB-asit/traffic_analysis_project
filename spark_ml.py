from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import os

spark = SparkSession.builder \
    .appName("UrbanMobility-MLPipeline") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
start_time = time.time()
os.makedirs("visualizations", exist_ok=True)

print("Loading dataset partitions from Parquet layer...")
data_df = spark.read.parquet("storage_layer/traffic_parquet")

print("\nBuilding SparkML Feature Pipeline Vectors...")
feature_cols = ["trip_distance", "passenger_count", "pickup_hour", "is_weekend"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
assembled_df = assembler.transform(data_df)

scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)

print("\nExecuting Distributed Train/Test Data Split (80/20)...")
train_data, test_data = scaled_df.randomSplit([0.8, 0.2], seed=42)

# ==========================================
# MODEL 1: Linear Regression (Predictive Pricing)
# ==========================================
print("\nInitializing SparkML Distributed Linear Regression Model...")
lr = LinearRegression(featuresCol="features", labelCol="fare_amount", predictionCol="predicted_fare")
lr_model = lr.fit(train_data)

print("\nModel Coefficients Evaluation Summary:")
for col_name, coef in zip(feature_cols, lr_model.coefficients):
    print(f" -> Feature '{col_name}': Weight Coefficient = {coef:.4f}")
print(f" -> Model Intercept Value: {lr_model.intercept:.4f}")

predictions = lr_model.transform(test_data)

evaluator_rmse = RegressionEvaluator(labelCol="fare_amount", predictionCol="predicted_fare", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="fare_amount", predictionCol="predicted_fare", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("\n" + "="*40)
print(f"FINAL MODEL PERFORMANCE METRICS:")
print(f"Root Mean Squared Error (RMSE): ${rmse:.4f}")
print(f"Coefficient of Determination (R^2): {r2:.4f}")
print("="*40)


# ==========================================
# MODEL 2: Random Forest (Feature Importance)
# ==========================================
print("\nTraining Distributed Random Forest for Feature Importance...")
rf = RandomForestRegressor(featuresCol="features", labelCol="fare_amount", numTrees=50, maxDepth=5)
rf_model = rf.fit(train_data)

importances = rf_model.featureImportances.toArray()

feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='magma', legend=False)
plt.title('Random Forest Feature Importance (Predicting Fare Amount)', fontsize=14)
plt.xlabel('Relative Importance (0.0 to 1.0)')
plt.ylabel('Engineered Features')
plt.tight_layout()
plt.savefig("visualizations/rf_feature_importance.png")
print("Saved ML visualization: visualizations/rf_feature_importance.png")

print(f"\nDistributed ML Training completed flawlessly in {time.time() - start_time:.2f} seconds!")

# Shut down the Spark engine ONLY after everything is completely done
spark.stop()
