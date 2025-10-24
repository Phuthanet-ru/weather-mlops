import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras import layers, models
import numpy as np
import os
import sys
import argparse

# หากคุณยังใช้ os.getcwd() หรือ os.path.isabs
# ในโค้ดอื่นที่ไม่แสดง ให้เพิ่ม import shutil, argparse หากจำเป็น

# mlflow.set_tracking_uri("file:./mlruns")
# ใช้ os.getcwd() แทน Path.cwd() หากคุณไม่ได้ import Path
# ใช้แค่ MLFLOW_TRACKING_URI เท่านั้น เนื่องจากไม่ได้ตั้งค่า MLFLOW_TRACKING_USERNAME/PASSWORD
REMOTE_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
if REMOTE_TRACKING_URI:
    mlflow.set_tracking_uri(REMOTE_TRACKING_URI)
else:
    mlflow.set_tracking_uri(f"file:{os.getcwd()}/mlruns")


def train_evaluate_register(preprocessing_run_id=None, epochs=10, lr=0.001):
    """เทรนโมเดลและบันทึกลง MLflow"""
    mlflow.set_experiment("Weather Classification - Model Training")

    data_path = "mlops_pipeline/data"

    # รับค่า epochs และ lr จาก Arguments หรือใช้ค่า Default
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=lr, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=epochs, help="Number of epochs")
    
    # ดึงค่า arguments ที่ส่งมาจาก command line (จาก GitHub Actions Matrix)
    if len(sys.argv) > 1:
        args, _ = parser.parse_known_args()
        lr = args.lr
        epochs = args.epochs
    
    # กำหนด run_name ตาม parameter ที่ถูกใช้จริง
    run_name = f"cnn_lr_{lr}_ep_{epochs}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("ml.step", "model_training_evaluation")

        IMG_SIZE = (128, 128)
        BATCH_SIZE = 32

        print(f"📂 โหลดข้อมูลจาก: {data_path}")
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_path,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_path,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
        )

        class_names = train_ds.class_names

        # Model Definition
        model = models.Sequential([
            layers.Rescaling(1./255),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(class_names), activation='softmax')
        ])

        # Model Compilation
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Model Training
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        # MLflow Logging
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])

        # MLflow Model Registration (ใช้ชื่อ artifact_path ที่สั้นลง)
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            input_example=np.zeros((1, 128, 128, 3)),
            registered_model_name="weather-classifier-prod"
        )

        print("✅ Training and logging completed successfully.")


if __name__ == "__main__":
    train_evaluate_register()
    