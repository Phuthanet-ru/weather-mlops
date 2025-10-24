import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os

mlflow.set_tracking_uri("file:./mlruns")

ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')


def remove_dot_files(root_dir):
    """ลบไฟล์ระบบหรือไฟล์ที่ไม่ใช่ภาพ"""
    count = 0
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(os.getcwd(), root_dir)
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for d in list(dirs):
            if d.startswith('.'):
                try:
                    os.rmdir(os.path.join(root, d))
                    dirs.remove(d)
                    count += 1
                except OSError:
                    pass
        for file in files:
            full_path = os.path.join(root, file)
            is_dot_file = file.startswith('.') or \
                file.lower() in ['thumbs.db', '.ds_store', 'desktop.ini']
            is_invalid_image = not file.lower().endswith(ALLOWED_EXTENSIONS)
            if is_dot_file or is_invalid_image:
                try:
                    os.remove(full_path)
                    count += 1
                except Exception as e:
                    print(f"ไม่สามารถลบไฟล์ {full_path}: {e}")
    return count


def remove_corrupted_images(root_dir):
    """ตรวจและลบรูปภาพที่เปิดไม่ได้ (ไฟล์เสีย)"""
    removed = 0
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(subdir, file)
            if file.lower().endswith(ALLOWED_EXTENSIONS):
                try:
                    img = Image.open(path)
                    img.verify()
                    img.close()
                except Exception:
                    print(f"🟥 พบไฟล์เสีย: {path} → ลบออก")
                    os.remove(path)
                    removed += 1
    return removed


def clean_non_images(root_dir):
    """ฟังก์ชันรวมการทำความสะอาดเบื้องต้น"""
    removed_dot = remove_dot_files(root_dir)
    removed_corrupt = remove_corrupted_images(root_dir)
    print(f"🧼 ลบไฟล์ระบบ {removed_dot} ไฟล์, "
          f"ลบไฟล์เสีย {removed_corrupt} ไฟล์")
    return removed_dot + removed_corrupt


def train_evaluate_register(preprocessing_run_id=None, epochs=10, lr=0.001):
    """เทรนโมเดลและบันทึกลง MLflow"""
    mlflow.set_experiment("Weather Classification - Model Training")

    data_path = "mlops_pipeline/data"
    clean_non_images(data_path)

    with mlflow.start_run(run_name=f"cnn_lr_{lr}_ep_{epochs}"):
        mlflow.set_tag("ml.step", "model_training_evaluation")

        IMG_SIZE = (128, 128)
        BATCH_SIZE = 32

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

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])

        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="weather_cnn_model",
            input_example=np.zeros((1, 128, 128, 3)),
            registered_model_name="weather-classifier-prod"
        )

        print("✅ Training and logging completed successfully.")


if __name__ == "__main__":
    train_evaluate_register()
