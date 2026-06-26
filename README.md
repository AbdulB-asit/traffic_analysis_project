# 🚕 Smart Urban Mobility & Traffic Analytics Engine

![Apache Spark](https://img.shields.io/badge/Apache%20Spark-F68A1E?style=for-the-badge&logo=apachespark&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Parquet](https://img.shields.io/badge/Apache_Parquet-000000?style=for-the-badge)

## 📌 Project Overview & The Need for Big Data

Urban traffic networks generate massive, continuous streams of data daily. Traditional row-based processing tools (like standard Pandas or Excel) suffer from severe memory bottlenecks and crashing when attempting to process these gigabyte-scale datasets on single machines.

This project was built to solve the **Volume** and **Velocity** challenges of urban transit data. It is a fully containerized, distributed Big Data pipeline that ingests millions of raw traffic records, optimizes them into a columnar storage format, and runs parallelized Machine Learning algorithms to predict transit fares and identify spatiotemporal bottlenecks.

By leveraging **Apache Spark**, this engine bypasses single-node RAM limitations via Lazy Evaluation and Resilient Distributed Datasets (RDDs).

---

## ⚙️ Core Architecture & Tech Stack

* **Infrastructure Layer:** Docker & Docker Compose (Isolated standalone Spark cluster).
* **Storage Layer:** Partitioned Apache Parquet (Columnar storage for optimized disk I/O).
* **Processing Layer:** PySpark DataFrames & Spark SQL.
* **Analytics & ML Layer:** SparkML (Native Distributed Machine Learning).
* **Visualization Layer:** Downsampled Pandas, Matplotlib, and Seaborn.

---

## 🧠 Big Data Core Concepts Applied

To demonstrate a true understanding of Big Data engineering, this project implements the following core paradigms across its pipeline:

### 1. The Genesis: Simulating Volume & Velocity (`data_generator.py`)

Standard systems choke trying to open 2,000,000 rows of traffic data. Instead of downloading a static file, this pipeline simulates a real-world "data firehose." It mathematically generates 2 million records using probability distributions to create realistic spatiotemporal data, including weighted morning and evening rush-hour spikes.

### 2. The Storage Layer: Columnar Partitioning (`1_ingest_preprocess.py`)

Machine learning cannot be efficiently executed on raw CSVs. CSV is a "row-based" format, requiring full table scans for metric aggregations. This pipeline engineers an optimized storage layer by converting the raw CSV into **Partitioned Apache Parquet** files. By shifting to columnar storage partitioned by `day_of_week`, queries scanning for specific days completely skip unrelated partitions, drastically reducing disk I/O.

### 3. Distributed Processing & EDA (`2_eda_analytics.py`)

Running massive group-by operations on local memory causes standard libraries to crash out of RAM. This engine utilizes **Spark SQL** and **Lazy Evaluation**. Instead of processing data in local memory, Spark maps the aggregations across parallel executors via a Directed Acyclic Graph (DAG). Data is only downsampled to local Pandas at the visualization layer *after* the Big Data workload is mathematically reduced.

### 4. Distributed Machine Learning (`spark_ml.py`)

Standard Machine Learning (like Scikit-Learn) expects the entire dataset to fit on one machine. This project utilizes native **SparkML** to train models on data scattered across the cluster:
* **VectorAssembler:** Formats multi-dimensional data into distributed memory arrays.
* **Linear Regression:** Trained across cluster nodes to predict trip fares based on spatial and temporal features.
* **Random Forest Regressor:** A parallelized ensemble method used mathematically to extract "Feature Importance," proving which variables (e.g., Trip Distance) actually drive transit pricing.

---

## 📊 Visual Insights

*(Note: These graphs are generated dynamically in the `/visualizations` folder during the EDA and ML pipeline execution).*

* **System-Wide Stress Heatmap:** Maps the highest-density pickup times across the 7-day week to identify infrastructure bottlenecks.
* **Feature Correlation Matrix:** A distributed 10% sample analysis proving feature independence to prevent model bias.
* **Random Forest Feature Importance:** Mathematical proof of feature weights driving the predictive pricing model.

---

## 💻 How to Run the Project Locally

To ensure complete reproducibility and to bypass "it works on my machine" host-OS errors, this project is fully containerized. You do not need to install Java, Hadoop, or Spark on your local machine.

### Prerequisites

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### Execution Steps

**1. Spin up the distributed infrastructure**
```bash
docker-compose up -d
```

**2. Access the Spark Master Node shell**
```bash
docker-compose exec spark-master bash
```

**3. Install Python dependencies inside the container**
```bash
cd /app
pip install -r requirements.txt
```

**4. Run the Pipeline in Sequential Order**
```bash
# Step 1: Generate the raw 2,000,000 row dataset
python data_generator.py

# Step 2: Ingest, clean, and write to Partitioned Parquet
spark-submit 1_ingest_preprocess.py

# Step 3: Run Distributed EDA and generate visualizations
spark-submit 2_eda_analytics.py

# Step 4: Train and evaluate the Distributed Machine Learning models
spark-submit spark_ml.py
```

### 🛑 Clean Up

To shut down the cluster and free up your system resources, simply exit the container shell (`exit`) and run the teardown command on your local machine:

```bash
docker-compose down
```
