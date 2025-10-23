import os
import tensorflow as tf
import mlflow
# 💡 กำหนดให้ MLflow ใช้โฟลเดอร์เก็บผล Artifacts ในเครื่องก่อนเสมอ
mlflow.set_tracking_uri("file:./mlruns")


def preprocess_data(data_dir="data", img_size=(128, 128), batch_size=32):
    """
    โหลดชุดข้อมูลภาพและเตรียม train/test split
    """
    mlflow.set_experiment("Weather Classification - Data Preprocessing")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step", "data_preprocessing")

        print(f"🚀 เริ่ม preprocessing (Run ID: {run_id})")

        # โหลด dataset จากโฟลเดอร์
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=img_size,
            batch_size=batch_size
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=img_size,
            batch_size=batch_size
        )

        mlflow.log_param("img_size", img_size)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_metric("train_batches", len(train_ds))
        mlflow.log_metric("val_batches", len(val_ds))

        print("✅ Data preprocessing สำเร็จแล้ว")

        # สำหรับ workflow ต่อเนื่องใน GitHub Actions
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                # บันทึก run_id เป็น output สำหรับ Job ต่อไป
                print(f"run_id={run_id}", file=f)

if __name__ == "__main__":
    preprocess_data("mlops_pipeline/data")