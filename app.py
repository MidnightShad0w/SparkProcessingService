from flask import Flask, request, jsonify
from spark_model_service import SparkModelService

app = Flask(__name__)

spark_service = SparkModelService(
    bucket_name="bucket-spark",
    minio_endpoint="http://localhost:9000",
    minio_access_key="minioadmin",
    minio_secret_key="minioadmin"
)


@app.route("/spark/api/process", methods=["POST"])
def process_csv_file():
    try:
        data = request.get_json()
        csv_path = data.get("filePath")
        if not csv_path:
            return jsonify({"error": "filePath not provided"}), 400

        model_path = "s3a://bucket-spark/model"
        result_path = "s3a://bucket-spark/result-anomaly-logs"

        spark_service.process_file(csv_path, model_path, result_path)

        return jsonify({"message": f"Модель успешно применена, результат сохранен в {result_path}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
