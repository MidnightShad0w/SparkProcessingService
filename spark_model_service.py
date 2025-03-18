from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id
import boto3
from botocore.client import Config
import numpy as np
import pandas as pd
import torch
import os

# Импортируем наши классы/функции для BERT+AutoEncoder
from ml_model import BertEncoder, AutoEncoder, compute_reconstruction_errors

from pyspark.sql.types import StructType, StructField, LongType, DoubleType


class SparkModelService:
    def __init__(
        self,
        bucket_name: str,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str
    ):
        self.bucket_name = bucket_name
        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key

        # 1) Инициализируем SparkSession, повторяя настройки из вашего log_anomaly_detection_bert.py
        self.spark = (
            SparkSession.builder
            .appName("AnomalyDetection")
            .master("local[*]")
            .config("spark.hadoop.fs.s3a.endpoint", self.minio_endpoint)
            .config("spark.hadoop.fs.s3a.access.key", self.minio_access_key)
            .config("spark.hadoop.fs.s3a.secret.key", self.minio_secret_key)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
            # Дополнительные настройки, аналогичные вашему рабочему скрипту
            .config("spark.network.timeout", "36000s")
            .config("spark.executor.heartbeatInterval", "3600s")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.python.worker.reuse", "false")
            .config("spark.local.ip", "127.0.0.1")
            .config(
                "spark.driver.extraJavaOptions",
                "--add-opens=java.base/java.nio=ALL-UNNAMED "
                + "--add-exports=java.base/java.nio=ALL-UNNAMED "
                + "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                + "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED"
            )
            .config(
                "spark.executor.extraJavaOptions",
                "--add-opens=java.base/java.nio=ALL-UNNAMED "
                + "--add-exports=java.base/java.nio=ALL-UNNAMED "
                + "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                + "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED"
            )
            .getOrCreate()
        )

        # 2) Инициализация клиента MinIO (S3) через boto3
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.minio_endpoint,
            aws_access_key_id=self.minio_access_key,
            aws_secret_access_key=self.minio_secret_key,
            config=Config(signature_version="s3v4"),
        )

        # 3) Загрузка обученной модели + threshold из MinIO (или локально)
        model_local_path = "/tmp/autoencoder.pt"
        threshold_local_path = "/tmp/threshold.txt"

        # Скачиваем модель и threshold
        self.s3_client.download_file(bucket_name, "model/autoencoder.pt", model_local_path)
        self.s3_client.download_file(bucket_name, "model/threshold.txt", threshold_local_path)

        # Инициализируем BERT + AutoEncoder
        self.device = "cpu"
        self.bert_encoder = BertEncoder(device=self.device)

        self.autoencoder = AutoEncoder()
        self.autoencoder.load_state_dict(torch.load(model_local_path, map_location=self.device))
        self.autoencoder.eval()

        # Загружаем threshold
        with open(threshold_local_path, "r", encoding="utf-8") as f:
            self.threshold = float(f.read().strip())

        print(">>> AutoEncoder и threshold загружены из MinIO.")

    def process_file(self, csv_path: str, model_path: str, result_path: str):
        """
        Аналог "теста" в log_anomaly_detection_bert.py:
          1) читаем CSV
          2) если есть _c0, убираем
          3) добавляем row_id
          4) collect() -> (row_id, Message)
          5) compute recon_error для каждого
          6) Spark DF => join => filter
          7) записываем => parquet
          8) удаляем исходные файлы
        """
        print("Путь к входному файлу:", csv_path)

        # 1) Читаем CSV
        try:
            input_df = (
                self.spark.read.option("header", "true")
                .option("inferSchema", "true")
                .csv(csv_path)
            )
        except Exception as e:
            if "Path does not exist" in str(e):
                print(f"Входной путь {csv_path} не существует. Возможно, все файлы уже обработаны.")
                return
            else:
                raise e

        # убираем _c0, если есть
        if "_c0" in input_df.columns:
            input_df = input_df.drop("_c0")

        input_df.printSchema()

        # 2) Добавляем row_id
        df_with_id = input_df.withColumn("row_id", monotonically_increasing_id())
        # собираем в Python-список
        rows = df_with_id.select("row_id", "Message").na.drop().collect()

        if not rows:
            print("Нет данных для обработки (нет Message).")
            return

        # row_id, Message
        row_ids = [int(r["row_id"]) for r in rows]
        messages = [r["Message"] for r in rows]

        # 3) Считаем reconstruction errors
        errors = compute_reconstruction_errors(
            texts=messages,
            bert_encoder=self.bert_encoder,
            autoencoder=self.autoencoder,
        )

        errors = [float(e) for e in errors]  # приведение к python float
        threshold = self.threshold
        print("Используем сохранённый threshold:", threshold)

        # 4) Формируем (row_id, recon_error) => Spark DF
        records = list(zip(row_ids, errors))
        schema = StructType(
            [
                StructField("row_id", LongType(), True),
                StructField("recon_error", DoubleType(), True),
            ]
        )
        error_df = self.spark.createDataFrame(records, schema=schema)

        # 5) JOIN c исходным DF по row_id
        joined_df = df_with_id.join(error_df, on="row_id", how="inner")

        # 6) Фильтруем аномалии
        anomalies_df = joined_df.filter(col("recon_error") > threshold)
        anomalies_df.show(20, False)

        # 7) Сохраняем (в parquet). Можем убрать row_id, если не нужен
        anomalies_df.distinct() \
            .coalesce(1) \
            .write \
            .mode("append") \
            .option("header", "true") \
            .csv(result_path)

        print(f"Результаты сохранены в: {result_path}")

        # 8) Удаляем исходные файлы
        self.delete_processed_files(prefix="uploads/")

    def ensure_folder_exists(self, folder_key: str):
        """
        Аналог создания "виртуальной папки" в MinIO (S3).
        """
        resp = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=folder_key)
        if "Contents" not in resp:
            self.s3_client.put_object(Bucket=self.bucket_name, Key=folder_key, Body=b"")
            print(f"Создана виртуальная папка: {folder_key}")

    def delete_processed_files(self, prefix="uploads/"):
        """
        Удаляем все файлы в заданном префиксе (uploads/), аналогично Java-коду.
        """
        try:
            resp = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if "Contents" in resp:
                for obj in resp["Contents"]:
                    key = obj["Key"]
                    self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
                    print(f"Deleted: {key}")
        except Exception as e:
            print("Ошибка при удалении файлов:", str(e))
