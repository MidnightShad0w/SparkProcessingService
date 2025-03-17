from airflow import DAG
from airflow.operators.http_operator import SimpleHttpOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='process_logs_dag',
    default_args=default_args,
    schedule_interval='*/1 * * * *',  # каждые 1 минуты
    catchup=False
) as dag:

    process_files = SimpleHttpOperator(
        task_id='process_csv_files',
        http_conn_id='spark_processing_conn',
        endpoint='/spark/api/process',
        method='POST',
        headers={"Content-Type": "application/json"},
        data='{"filePath": "s3a://bucket-spark/uploads/"}',
    )

    process_files
