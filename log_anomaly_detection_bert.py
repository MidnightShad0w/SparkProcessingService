import os
import re
import numpy as np
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, udf
from pyspark.sql.types import StructType, StructField, LongType, DoubleType, StringType, FloatType

# Импорт из ml_model
from ml_model import (
    BertEncoder,
    AutoEncoder,
    train_autoencoder,
    compute_reconstruction_errors
)


def normalize_message(msg):
    """
    - Заменяем IP-адреса -> [IP]
    - Заменяем User\d+ -> [USER]
    - Прочие числа, если хотим, можно заменить -> [NUM], но здесь оставим.
    """
    msg = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]', str(msg))
    msg = re.sub(r'\bUser\d+\b', '[USER]', msg)
    return msg


def parse_time_taken(s):
    """
    '28ms' -> 28.0
    """
    if isinstance(s, str) and s.endswith("ms"):
        s = s[:-2]
    try:
        return float(s)
    except:
        return 0.0


def combine_fields_for_bert(loglevel, service, message, timestr=None):
    """
    Объединяем LogLevel, Service и сам текст лог-сообщения.
    Например: "LOGLEVEL=WARNING SERVICE=ServiceA TEXT=Performance Warnings"
    Или любой другой формат, лишь бы BERT понимал в 1 строке.
    """
    # Можно дополнительно нормализовать IP, User... при желании
    # Здесь упрощённо берём всё "как есть"
    if timestr is not None:
        return f"LOGLEVEL={loglevel} SERVICE={service} TIME={timestr} TEXT={message}"
    else:
        return f"LOGLEVEL={loglevel} SERVICE={service} TEXT={message}"


# ==== Основная логика ====
def main():
    logDataPath = r"C:\Users\admin\Desktop\Diplom\SparkProcessingService\data\partlogdata1.csv"
    modelSaveDir = r"C:\Users\admin\Desktop\Diplom\SparkProcessingService\model"
    anomaliesSavePath = r"C:\Users\admin\Desktop\Diplom\SparkProcessingService\output\anomaly-logdata"

    # Инициализация SparkSession (как у вас в коде)
    spark = SparkSession.builder \
        .appName("LogAnomalyDetectionBERT") \
        .master("local[*]") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.network.timeout", "36000s") \
        .config("spark.executor.heartbeatInterval", "3600s") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.python.worker.reuse", "false") \
        .config("spark.local.ip", "127.0.0.1") \
        .config("spark.driver.extraJavaOptions",
                "--add-opens=java.base/java.nio=ALL-UNNAMED " +
                "--add-exports=java.base/java.nio=ALL-UNNAMED " +
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED " +
                "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED") \
        .config("spark.executor.extraJavaOptions",
                "--add-opens=java.base/java.nio=ALL-UNNAMED " +
                "--add-exports=java.base/java.nio=ALL-UNNAMED " +
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED " +
                "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED") \
        .getOrCreate()

    print(spark.version)

    # Чтение CSV
    logs = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(logDataPath)

    # Если лишняя колонка _c0:
    if "_c0" in logs.columns:
        logs = logs.drop("_c0")

    # Определим Spark UDF
    normalize_message_udf = udf(normalize_message, StringType())
    parse_time_udf = udf(parse_time_taken, FloatType())

    # 2) Применяем UDF к DataFrame
    logs = logs.withColumn("Message", normalize_message_udf(col("Message")))
    logs = logs.withColumn("TimeVal", parse_time_udf(col("TimeTaken")))

    logs.printSchema()

    # Разделяем на train/test
    train_df, test_df = logs.randomSplit([0.7, 0.3], seed=42)

    # == TRAIN ==
    # Добавим row_id для удобства
    train_df = train_df.withColumn("row_id", monotonically_increasing_id())
    # Собираем поля, которые нужны:
    # row_id, LogLevel, Service, Message
    # (na.drop в случае, если Message пустое)
    train_rows = train_df.select("row_id", "LogLevel", "Service", "Message", "TimeVal").na.drop().collect()
    train_ids = [r["row_id"] for r in train_rows]

    # Объединяем LogLevel+Service+Message для BERT
    train_combined_texts = [
        combine_fields_for_bert(
            r["LogLevel"],
            r["Service"],
            r["Message"],
            timestr=str(r["TimeVal"])
        )
        for r in train_rows
    ]

    # Инициализируем BERT и AE
    device = "cpu"
    bert_encoder = BertEncoder(device=device)
    autoencoder = AutoEncoder().to(device)

    print("=== Training AutoEncoder ===")
    # Обучаем
    train_autoencoder(
        texts=train_combined_texts,
        bert_encoder=bert_encoder,
        autoencoder=autoencoder,
        num_epochs=3,  # можно увеличить
        batch_size=32,
        device=device
    )

    print("=== Compute errors on train ===")
    train_errors = compute_reconstruction_errors(
        texts=train_combined_texts,
        bert_encoder=bert_encoder,
        autoencoder=autoencoder,
        device=device,
        batch_size=32
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
    test_rows = test_df.select("row_id", "LogLevel", "Service", "Message", "TimeVal").na.drop().collect()

    test_ids = [int(r["row_id"]) for r in test_rows]
    test_combined_texts = [
        combine_fields_for_bert(
            r["LogLevel"],
            r["Service"],
            r["Message"],
            timestr=str(r["TimeVal"])
        )
        for r in test_rows
    ]

    # Вычисляем reconstruction errors
    test_errors = compute_reconstruction_errors(
        texts=test_combined_texts,
        bert_encoder=bert_encoder,
        autoencoder=autoencoder,
        device=device,
        batch_size=32
    )

    # Преобразуем значения в float
    test_errors = [float(e) for e in test_errors]

    # Создаём список (row_id, recon_error)
    records = list(zip(test_ids, test_errors))

    # Определяем схему
    schema = StructType([
        StructField("row_id", LongType(), True),
        StructField("recon_error", DoubleType(), True)
    ])

    # Создаём Spark DataFrame
    test_result_spark = spark.createDataFrame(records, schema=schema)

    # JOIN
    joined_test_df = test_df.join(test_result_spark, on="row_id", how="inner")
    anomalies_df = joined_test_df.filter(col("recon_error") > threshold)

    count_anom = anomalies_df.count()
    print(f"Found {count_anom} anomalies in test set.")
    anomalies_df.show(10, False)

    # Сохраняем аномалии в один CSV
    anomalies_df.drop("row_id").coalesce(1).write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(anomaliesSavePath)

    print(f"Anomalies CSV saved to: {anomaliesSavePath}")
    print("Spark UI (if local) at http://localhost:4040")
    print("Нажмите ENTER для завершения.")
    input()
    spark.stop()


if __name__ == "__main__":
    main()
