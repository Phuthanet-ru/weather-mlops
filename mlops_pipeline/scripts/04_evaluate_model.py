import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# กำหนด tracking URI ให้ MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Weather Classification - Model Evaluation")

MODEL_NAME = "weather-classifier-prod"
model_stage = "Staging"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DATA_PATH = "mlops_pipeline/data"

def evaluate_model():

    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    if os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true" # หากใช้ Self-signed certs
        mlflow.set_experiment("Weather Classification - Model Evaluation") # เรียก set_experiment หลัง set_tracking_uri

    print("📦 กำลังโหลดโมเดลจาก MLflow Registry...")
    model_uri = f"models:/{MODEL_NAME}/{model_stage}"
    model = mlflow.tensorflow.load_model(model_uri)

    print("📂 โหลดข้อมูล Validation/Test Set...")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = test_ds.class_names
    y_true, y_pred = [], []

    print("🧠 ประเมินโมเดล...")
    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"✅ Test Accuracy: {acc:.4f}")

    # Log metrics
    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_metric("test_accuracy", acc)

        # สร้าง confusion matrix แล้ว log เป็น artifact
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(xticks_rotation=45)
        os.makedirs("evaluation_artifacts", exist_ok=True)
        cm_path = "evaluation_artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

    print("📊 ประเมินโมเดลเสร็จสิ้น และบันทึกผลใน MLflow แล้ว!")

if __name__ == "__main__":
    evaluate_model()
