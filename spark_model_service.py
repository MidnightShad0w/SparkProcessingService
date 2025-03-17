from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType
import boto3
from botocore.client import Config
import numpy as np

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

        # Инициализируем SparkSession
        self.spark = SparkSession.builder \
            .appName("AnomalyDetection") \
            .master("local[*]") \
            .config("spark.hadoop.fs.s3a.endpoint", self.minio_endpoint) \
            .config("spark.hadoop.fs.s3a.access.key", self.minio_access_key) \
            .config("spark.hadoop.fs.s3a.secret.key", self.minio_secret_key) \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .getOrCreate()

        # Инициализация клиента MinIO (S3) через boto3
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.minio_endpoint,
            aws_access_key_id=self.minio_access_key,
            aws_secret_access_key=self.minio_secret_key,
            config=Config(signature_version='s3v4')
        )

        # Для примера создадим объекты BERT-энкодера и загруженный автоэнкодер
        # Предположим, что мы уже где-то обучили автоэнкодер и просто подгружаем его веса.
        # Или же используем его напрямую без сложного сохранения, как пример.
        self.bert_encoder = BertEncoder()
        self.autoencoder = AutoEncoder()
        # На практике: autoencoder.load_state_dict(torch.load("path_to_autoencoder_weights.pth"))

    def process_file(self, csv_path: str, model_path: str, result_path: str):
        """
        csv_path   – путь к входному CSV (s3a://bucket-spark/uploads/....csv)
        model_path – формально путь к папке модели (для совместимости с Java-кодом),
                     но здесь мы модель берём напрямую в коде.
        result_path – путь, куда сохранить parquet с аномалиями (s3a://bucket-spark/result-anomaly-logs).
        """
        print("Путь к входному файлу:", csv_path)

        # 1) Читаем CSV
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

        input_df.printSchema()

        # Предположим, что логи лежат в поле "log_text" (проверьте реальное имя колонки!)
        # Соберём все тексты в локальный список (с осторожностью, если очень большой датасет!)
        # Для демонстрации берём .collect() – на больших данных нужен другой подход.
        log_rows = input_df.select("log_text").rdd.map(lambda row: row[0]).collect()
        # Некоторые поля могут быть None. Отфильтруем:
        log_rows = [text for text in log_rows if text is not None]

        # 2) Считаем ошибки реконструкции для каждого лога
        errors = compute_reconstruction_errors(
            texts=log_rows,
            bert_encoder=self.bert_encoder,
            autoencoder=self.autoencoder,
            device="cpu"
        )

        # 3) Определяем порог аномалии, например, 90-й перцентиль
        threshold = float(np.percentile(errors, 90))
        print("Anomaly threshold (90th percentile):", threshold)

        # 4) Превращаем список ошибок обратно в DataFrame для объединения
        # Создадим вспомогательный DF: каждая строка: (log_text, reconstruction_error)
        error_tuples = list(zip(log_rows, errors))  # [(text, err), ...]
        error_df = self.spark.createDataFrame(error_tuples, ["log_text", "recon_error"])

        # 5) Объединим error_df с input_df по "log_text" (упрощённо – JOIN):
        joined_df = input_df.join(error_df, on="log_text", how="inner")

        # 6) Выбираем строки с аномалиями
        anomalies_df = joined_df.filter(col("recon_error") > threshold)
        anomalies_df.show(20, False)

        # 7) Создаём папку, если нужно
        self.ensure_folder_exists("result-anomaly-logs/")

        # 8) Сохраняем результат
        anomalies_df.write.mode("append").parquet(result_path)
        print(f"Результаты сохранены в: {result_path}")

        # 9) Удаляем обработанные файлы
        self.delete_processed_files(prefix="uploads/")

    def ensure_folder_exists(self, folder_key: str):
        """
        Аналог создания "виртуальной папки" в MinIO (S3).
        """
        # Проверяем, есть ли объект с таким префиксом
        resp = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=folder_key)
        if 'Contents' not in resp:
            # Создаём пустой объект
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
