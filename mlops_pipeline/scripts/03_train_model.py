import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os
from pathlib import Path
import argparse
import shutil #  🚨  เพิ่ม import shutil สำหรับการลบ
#  directory ที่ไม่ใช่ directory ว่าง


# 💡 บันทึกค่า Remote Tracking URI (จาก Environment Variables)
# หากมีการตั้งค่าไว้ใน GitHub Actions
REMOTE_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

# 💡 กำหนดให้ MLflow ใช้โฟลเดอร์เก็บผล Artifacts ในเครื่องก่อนเสมอ
mlflow.set_tracking_uri(f"file:{Path.cwd()}/mlruns")

ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')


def remove_dot_files(root_dir):
    """ลบไฟล์ระบบหรือไฟล์ที่ไม่ใช่ภาพ"""
    count = 0
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(os.getcwd(), root_dir)
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for d in list(dirs):
            if d.startswith('.'):
                full_path = os.path.join(root, d)
                #  💡 ต้องกำหนด full_path ก่อนลบ
                try:
                    #  🚨 แก้ไข: ใช้ shutil.rmtree แทน
                    # os.rmdir เพื่อลบ directory ที่ซ่อนอยู่
                    shutil.rmtree(full_path)
                    dirs.remove(d)
                    count += 1
                except OSError:
                    # ใน test case ของคุณ โฟลเดอร์ว่างและควรลบได้
                    # แต่ถ้ามีไฟล์อยู่ shutil.rmtree จะทำงาน
                    pass
            for file in files:
                full_path = os.path.join(root, file)
                # E501 fix: ตัดบรรทัดให้สั้นลง
                is_dot_file = file.startswith('.') or \
                    file.lower() in ['thumbs.db', '.ds_store', 'desktop.ini']
                # E501 fix: ตัดบรรทัดให้สั้นลง
                is_invalid_image = not file.lower().endswith(
                    ALLOWED_EXTENSIONS)
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

    # 💡 หากมี Remote URI ให้ Log Metadata ไปที่ Remote Server ด้วย
    # E501 fix: ตัดบรรทัด
    # การ set เป็น Remote อีกครั้งในฟังก์ชันนี้ จะทำให้ Log Metadata
    # ไปที่ Remote
    # แต่ Artifacts (โมเดล) จะถูกบันทึกใน Local ก่อนแล้วค่อย sync ไป remote
    if REMOTE_TRACKING_URI:
        mlflow.set_tracking_uri(REMOTE_TRACKING_URI)

    # 💡 ใช้ชื่อ Artifact Path ที่ชัดเจน และจะใช้ในการค้นหาใน main.yml
    ARTIFACT_PATH = "model"

    with mlflow.start_run(run_name=f"cnn_lr_{lr}_ep_{epochs}"):
        mlflow.set_tag("ml.step", "model_training_evaluation")
        if preprocessing_run_id:
            mlflow.log_param("preprocessing_run_id", preprocessing_run_id)

        IMG_SIZE = (128, 128)
        BATCH_SIZE = 32
        data_path = "mlops_pipeline/data"

        # Data Validation
        cleaned_count = remove_dot_files(data_path)
        corrupted_count = remove_corrupted_images(data_path)
        # E501 fix
        print(f"🧼 ลบไฟล์ระบบ {cleaned_count} ไฟล์, "
              f"ลบไฟล์เสีย {corrupted_count} ไฟล์")

        # 💡 สร้างและบันทึกรายงาน Data Validation Artifacts
        # 💡 แก้ E501 และ E128
        validation_status = ('PASS' if cleaned_count + corrupted_count == 0
                             else 'WARNING')

        report_content = (
            f"--- Data Validation Report ---\n"
            f"Total files removed (system/invalid): {cleaned_count}\n"
            f"Total corrupted images removed: {corrupted_count}\n"
            # E501 fix: ใช้ตัวแปร status
            f"Validation Check Status: {validation_status}\n"
            f"------------------------------\n"
        )

        report_file = "data_validation_report.txt"
        with open(report_file, "w") as f:
            f.write(report_content)

        mlflow.log_artifact(report_file, "data_validation")
        # F541 fix: ลบ f-string ที่ไม่มี placeholder
        print("✅ Logged Data Validation Report to MLflow Artifacts.")
        os.remove(report_file)

        # Data Loading and Preprocessing
        print(f"📂 โหลดข้อมูลจาก: {data_path}")
        temp_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_path, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
        class_names = temp_ds.class_names

        if len(class_names) < 2:
            # E501 fix
            raise ValueError(
                f"⚠️ Dataset ต้องมีอย่างน้อย 2 classes แต่พบเพียง "
                f"{len(class_names)}: {class_names}")

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_path, validation_split=0.2, subset="training", seed=42,
            # E501 fix
            image_size=IMG_SIZE, batch_size=BATCH_SIZE, labels='inferred',
            label_mode='int'
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_path, validation_split=0.2, subset="validation", seed=42,
            # E501 fix
            image_size=IMG_SIZE, batch_size=BATCH_SIZE, labels='inferred',
            label_mode='int'
        )

        # Data Augmentation & Rescaling (Pre-processing)
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        # Model Definition and Training
        model = models.Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            # E231 fix
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            # E231 fix
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(class_names), activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      # E501 fix
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        # MLflow Tracking and Registration
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])

        # 1. Log Model (บันทึกไฟล์ Artifacts ลงใน Local Disk)
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path=ARTIFACT_PATH,
            input_example=np.zeros((1, 128, 128, 3)),
            registered_model_name=None
        )

        # 2. Register Model (ใช้ URI ที่ถูกต้อง)
        val_acc = history.history["val_accuracy"][-1]
        if val_acc >= 0.60:
            run_id = mlflow.active_run().info.run_id
            # ใช้ URI ที่ MLflow มองเห็น (Local Path ที่ถูกสร้าง)
            model_uri = f"runs:/{run_id}/{ARTIFACT_PATH}"
            print(f"🔗 Registering model from URI: {model_uri}")

            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name="weather-classifier-prod"
            )
            print(f"✅ Registered model version: {registered_model.version}")
        else:
            print(f"⚠️ Accuracy {val_acc:.2f} ต่ำกว่าเกณฑ์ ไม่ลงทะเบียนโมเดล")


# E305 fix: เพิ่ม 2 บรรทัดว่างเปล่า
if __name__ == "__main__":

    # E501 fix
    parser = argparse.ArgumentParser(
        description="Run model training and evaluation.")

    # E501 fix
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train.")

    # E501 fix
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate for the optimizer.")

    args = parser.parse_args()

    train_evaluate_register(epochs=args.epochs, lr=args.lr)
# W292 fix: เพิ่มบรรทัดว่างเปล่าที่ท้ายไฟล์
