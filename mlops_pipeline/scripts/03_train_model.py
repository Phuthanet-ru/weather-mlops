import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras import layers, models
from PIL import Image
import os
from pathlib import Path

# กำหนดให้ MLflow ใช้โฟลเดอร์เก็บผลในเครื่อง
mlflow.set_tracking_uri("file:./mlruns")


# ✅ นามสกุลภาพที่อนุญาต
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
            is_dot_file = file.startswith('.') or file.lower() in ['thumbs.db', '.ds_store', 'desktop.ini']
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
                except Exception:
                    print(f"🟥 พบไฟล์เสีย: {path} → ลบออก")
                    os.remove(path)
                    removed += 1
    return removed


def train_evaluate_register(preprocessing_run_id=None, epochs=10, lr=0.001):
    mlflow.set_experiment("Weather Classification - Model Training")

    with mlflow.start_run(run_name=f"cnn_lr_{lr}_ep_{epochs}"):
        mlflow.set_tag("ml.step", "model_training_evaluation")
        if preprocessing_run_id:
            mlflow.log_param("preprocessing_run_id", preprocessing_run_id)

        IMG_SIZE = (128, 128)
        BATCH_SIZE = 32
        data_path = "mlops_pipeline/data"  # ✅ โฟลเดอร์เก็บภาพ

        # -----------------------------------------------
        # 🧹 ทำความสะอาดข้อมูลก่อนโหลด
        cleaned_count = remove_dot_files(data_path)
        corrupted_count = remove_corrupted_images(data_path)
        print(f"🧼 ลบไฟล์ระบบ {cleaned_count} ไฟล์, ลบไฟล์เสีย {corrupted_count} ไฟล์")
        # -----------------------------------------------

        print(f"📂 โหลดข้อมูลจาก: {data_path}")

        # ✅ โหลด dataset ครั้งแรกเพื่ออ่าน class names
        temp_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_path,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
        class_names = temp_ds.class_names

        if len(class_names) < 2:
            raise ValueError(f"⚠️ Dataset ต้องมีอย่างน้อย 2 classes แต่พบเพียง {len(class_names)}: {class_names}")

        # ✅ สร้าง train และ validation set
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_path,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            labels='inferred',
            label_mode='int'
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_path,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            labels='inferred',
            label_mode='int'
        )

        # ✅ Data augmentation
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        # ✅ Model architecture
        model = models.Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(32, (3,3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(class_names), activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # ✅ Train
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        # ✅ Log parameters และ metrics
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])

        # ✅ Log model
        mlflow.tensorflow.log_model(model, "weather_cnn_model")

        # ✅ ตรวจสอบเกณฑ์เพื่อ register model
        val_acc = history.history["val_accuracy"][-1]
        if val_acc >= 0.60:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/weather_cnn_model"
            registered_model = mlflow.register_model(model_uri, "weather-classifier-prod")
            print(f"✅ Registered model version: {registered_model.version}")
        else:
            print(f"⚠️ Accuracy {val_acc:.2f} ต่ำกว่าเกณฑ์ ไม่ลงทะเบียนโมเดล")


if __name__ == "__main__":
    train_evaluate_register()
