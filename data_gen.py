import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Generating 2,000,000 rows of highly realistic synthetic traffic data...")
n_rows = 2000000

# We map out 24 hours. The higher the number, the more traffic that hour gets.
# Notice the massive spikes at index 8 (8 AM) and index 17 (5 PM).
hour_weights = [1, 1, 1, 1, 1, 2, 8, 12, 10, 7, 5, 5, 5, 5, 6, 7, 10, 15, 12, 8, 5, 4, 3, 2]
hour_probs = np.array(hour_weights) / sum(hour_weights) # Normalize to percentages

# Generate realistic temporal distribution
random_days = np.random.randint(0, 30, n_rows)
random_hours = np.random.choice(np.arange(24), size=n_rows, p=hour_probs)
random_minutes = np.random.randint(0, 60, n_rows)

# Fast vectorized timestamp generation
timestamps = pd.to_datetime('2026-01-01') + pd.to_timedelta(random_days, unit='d') + \
             pd.to_timedelta(random_hours, unit='h') + pd.to_timedelta(random_minutes, unit='m')

data = {
    "trip_id": np.arange(1000000, 1000000 + n_rows),
    "pickup_datetime": timestamps,
    "pickup_zone_id": np.random.randint(1, 264, n_rows),
    "dropoff_zone_id": np.random.randint(1, 264, n_rows),
    "passenger_count": np.random.choice([1, 2, 3, 4, 5], n_rows, p=[0.7, 0.15, 0.05, 0.05, 0.05]),
    "trip_distance": np.round(np.random.exponential(scale=3.0, size=n_rows) + 0.5, 2),
    "fare_amount": np.zeros(n_rows),
    "tip_amount": np.round(np.random.exponential(scale=1.5, size=n_rows), 2)
}

# Base fare calculation formula plus noise
data["fare_amount"] = np.round(2.5 + (data["trip_distance"] * 2.0) + np.random.normal(0, 1.0, n_rows), 2)
data["fare_amount"] = np.clip(data["fare_amount"], 2.5, None)

df = pd.DataFrame(data)
df.to_csv("raw_traffic_data.csv", index=False)
print("Successfully generated raw_traffic_data.csv!")
