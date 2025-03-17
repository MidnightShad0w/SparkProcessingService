from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import boto3
from botocore.client import Config
import numpy as np
import pandas as pd
import torch
import os

# Импортируем наши классы/функции для BERT+AutoEncoder
from ml_model import BertEncoder, AutoEncoder, compute_reconstruction_errors


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

        # 1) Инициализируем SparkSession
        self.spark = SparkSession.builder \
            .appName("AnomalyDetection") \
            .master("local[*]") \
            .config("spark.hadoop.fs.s3a.endpoint", self.minio_endpoint) \
            .config("spark.hadoop.fs.s3a.access.key", self.minio_access_key) \
            .config("spark.hadoop.fs.s3a.secret.key", self.minio_secret_key) \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1") \
            .getOrCreate()

        # 2) Инициализация клиента MinIO (S3) через boto3
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.minio_endpoint,
            aws_access_key_id=self.minio_access_key,
            aws_secret_access_key=self.minio_secret_key,
            config=Config(signature_version='s3v4')
        )

        # 3) Загрузка обученной модели + threshold из MinIO (или локально)
        #    Предположим, что в MinIO в папке model/ лежат автоэнкодер и threshold.
        #    Скачаем их на локальную машину (в папку /tmp) и загрузим в память.
        model_local_path = "/tmp/autoencoder.pt"
        threshold_local_path = "/tmp/threshold.txt"

        # Скачиваем из MinIO в локальные файлы. Если у вас S3-совместимое хранилище,
        # то объект называется "model/autoencoder.pt", "model/threshold.txt".
        self.s3_client.download_file(bucket_name, "model/autoencoder.pt", model_local_path)
        self.s3_client.download_file(bucket_name, "model/threshold.txt", threshold_local_path)

        # Инициализируем BERT + AutoEncoder
        self.device = "cpu"  # или "cpu"
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
        csv_path   – путь к входному CSV (например, s3a://bucket-spark/uploads/something.csv)
        model_path – формально путь к папке модели (для совместимости),
                     но фактически мы модель уже загрузили в __init__.
        result_path – куда сохранить parquet с аномалиями (s3a://bucket-spark/result-anomaly-logs).
        """
        print("Путь к входному файлу:", csv_path)

        # 1) Читаем CSV.
        try:
            input_df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(csv_path)
        except Exception as e:
            if "Path does not exist" in str(e):
                print(f"Входной путь {csv_path} не существует. Возможно, все файлы уже обработаны.")
                return
            else:
                raise e

        # Если ваш CSV имеет лишнюю колонку _c0, можно её дропнуть:
        if "_c0" in input_df.columns:
            input_df = input_df.drop("_c0")

        input_df.printSchema()

        # 2) Собираем тексты. (Это дорого при больших объёмах!)
        rows = input_df.select("Message").na.drop().collect()
        messages = [r["Message"] for r in rows]

        if not messages:
            print("Нет текстов для обработки (колонка Message пуста?).")
            return

        # 3) Считаем reconstruction error для каждого сообщения
        errors = compute_reconstruction_errors(
            texts=messages,
            bert_encoder=self.bert_encoder,
            autoencoder=self.autoencoder
        )

        # Не нужен percentil, т.к. у нас уже есть self.threshold,
        # которую мы загрузили из обучающей стадии.
        threshold = self.threshold
        print("Используем сохранённый threshold:", threshold)

        # 4) Собираем (Message, recon_error) в Pandas => Spark
        pdf = pd.DataFrame({"Message": messages, "recon_error": errors})
        error_df = self.spark.createDataFrame(pdf)

        # 5) JOIN по Message
        joined_df = input_df.join(error_df, on="Message", how="inner")

        # 6) Фильтруем аномалии
        anomalies_df = joined_df.filter(col("recon_error") > threshold)

        anomalies_df.show(20, False)

        # 7) Создаём папку, если надо
        self.ensure_folder_exists("result-anomaly-logs/")

        # 8) Сохраняем аномалии
        anomalies_df.write.mode("append").parquet(result_path)
        print(f"Результаты сохранены в: {result_path}")

        # 9) Удаляем исходные файлы из uploads/
        self.delete_processed_files(prefix="uploads/")

    def ensure_folder_exists(self, folder_key: str):
        """
        Аналог создания "виртуальной папки" в MinIO (S3).
        """
        resp = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=folder_key)
        if 'Contents' not in resp:
            self.s3_client.put_object(Bucket=self.bucket_name, Key=folder_key, Body=b'')
            print(f"Создана виртуальная папка: {folder_key}")

    def delete_processed_files(self, prefix="uploads/"):
        """
        Удаляем все файлы в заданном префиксе (uploads/), аналогично Java-коду.
        """
        try:
            resp = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if 'Contents' in resp:
                for obj in resp['Contents']:
                    key = obj['Key']
                    self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
                    print(f"Deleted: {key}")
        except Exception as e:
            print("Ошибка при удалении файлов:", str(e))
