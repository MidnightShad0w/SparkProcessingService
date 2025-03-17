import os
import numpy as np
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, LongType, DoubleType

# Импорт из ml_model
from ml_model import (
    BertEncoder,
    AutoEncoder,
    train_autoencoder,
    compute_reconstruction_errors
)

# ==== Конфигурационные пути (адаптируйте под себя) ====
logDataPath = r"Z:\Diplom\SparkProcessingService\data\partlogdata1.csv"
modelSaveDir = r"Z:\Diplom\SparkProcessingService\model"
anomaliesSavePath = r"Z:\Diplom\SparkProcessingService\output\anomaly-logdata"

# ==== Инициализация SparkSession ====
spark = SparkSession.builder \
    .appName("LogAnomalyDetectionBERT") \
    .master("local[*]") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

# ==== Чтение CSV ====
logs = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(logDataPath)

# Если в CSV есть лишняя колонку _c0 из-за пустого заголовка - убираем её
if "_c0" in logs.columns:
    logs = logs.drop("_c0")

logs.printSchema()

# ==== Делим на train/test ====
train_df, test_df = logs.randomSplit([0.7, 0.3], seed=42)

# == TRAIN ==
train_df = train_df.withColumn("row_id", monotonically_increasing_id())
train_rows = train_df.select("row_id", "Message").na.drop().collect()
train_ids = [r["row_id"] for r in train_rows]
train_messages = [r["Message"] for r in train_rows]

device = "cpu"  # или "cpu"
bert_encoder = BertEncoder(device=device)
autoencoder = AutoEncoder().to(device)

print("=== Training AutoEncoder ===")
train_autoencoder(
    texts=train_messages,
    bert_encoder=bert_encoder,
    autoencoder=autoencoder,
    num_epochs=3,
    batch_size=32,
    device=device
)

print("=== Compute errors on train ===")
train_errors = compute_reconstruction_errors(
    train_messages,
    bert_encoder,
    autoencoder
)
threshold = float(np.percentile(train_errors, 90))
print("Anomaly threshold (90th percentile):", threshold)

# Сохраняем веса
os.makedirs(modelSaveDir, exist_ok=True)
torch.save(autoencoder.state_dict(), os.path.join(modelSaveDir, "autoencoder.pt"))
with open(os.path.join(modelSaveDir, "threshold.txt"), "w") as f:
    f.write(str(threshold))

# == TEST ==
print("=== Compute errors on test ===")
test_df = test_df.withColumn("row_id", monotonically_increasing_id())
test_rows = test_df.select("row_id", "Message").na.drop().collect()

# Приводим row_id к int, чтобы не тащить numpy типы
test_ids = [int(r["row_id"]) for r in test_rows]
test_messages = [r["Message"] for r in test_rows]

# Вычисляем reconstruction errors для тестового набора
test_errors = compute_reconstruction_errors(
    test_messages,
    bert_encoder,
    autoencoder
)

# Преобразуем значения в Python float
test_errors = [float(e) for e in test_errors]

# 1) Создаём список кортежей (row_id, recon_error)
records = list(zip(test_ids, test_errors))

# 2) Определяем схему
schema = StructType([
    StructField("row_id", LongType(), True),
    StructField("recon_error", DoubleType(), True)
])

# 3) Создаём Spark DataFrame из list of tuples
test_result_spark = spark.createDataFrame(records, schema=schema)

# JOIN
joined_test_df = test_df.join(test_result_spark, on="row_id", how="inner")
anomalies_df = joined_test_df.filter(col("recon_error") > threshold)

count_anom = anomalies_df.count()
print(f"Found {count_anom} anomalies in test set.")
anomalies_df.show(10, False)

anomalies_df.drop("row_id").coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(anomaliesSavePath)

print(f"Anomalies CSV saved to: {anomaliesSavePath}")
print("Spark UI (if local) at http://localhost:4040")
print("Нажмите ENTER для завершения.")
input()
spark.stop()
