import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys  # 💡 นำเข้า sys สำหรับการรับ Argument
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- กำหนดค่าคงที่ (ส่วนนี้จะไม่เปลี่ยนแปลง) ---
MODEL_NAME = "weather-classifier-prod"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DATA_PATH = "mlops_pipeline/data"
THRESHOLD = 0.60 # 💡 กำหนดเกณฑ์ความแม่นยำสำหรับการย้าย Stage

def evaluate_and_transition_model():

    # 1. 💡 จัดการ Argument: รับชื่อโมเดลและ Stage/Version ที่ต้องการโหลด
    if len(sys.argv) > 1:
        model_name_to_load = sys.argv[1] # 'weather-classifier-prod'
    else:
        model_name_to_load = MODEL_NAME
    
    # กำหนด Stage/Version ที่จะโหลด: ใช้ 'Latest' เสมอเมื่อเรียกจาก Pipeline (Transition)
    # หรือใช้ 'Staging' ถ้าต้องการประเมินเฉพาะโมเดลใน Staging
    if len(sys.argv) > 2:
        model_stage_to_load = sys.argv[2] # 'Latest'
    else:
        model_stage_to_load = "Latest" # Default ให้โหลด Latest เสมอสำหรับการประเมิน

    # 2. 💡 กำหนดค่า MLflow Client
    # ลบการตั้งค่า local 'file:./mlruns' ออก เพราะเราต้องการใช้ Environment Variables ใน CI/CD
    
    # MLflow จะใช้ ENV VARS (MLFLOW_TRACKING_URI, USERNAME, PASSWORD) โดยอัตโนมัติ
    # เราจึงแค่เรียก set_experiment
    try:
        mlflow.set_experiment("Weather Classification - Model Evaluation")
    except Exception as e:
        print(f"⚠️ Warning: Could not set MLflow experiment. Check tracking URI. Error: {e}")
        
    # --- เริ่มโหลดและประเมินโมเดล ---
    
    print(f"📦 กำลังโหลดโมเดล: {model_name_to_load} Stage: {model_stage_to_load} จาก MLflow Registry...")
    
    # 💡 ใช้ Stage/Version ที่รับเข้ามา
    model_uri = f"models:/{model_name_to_load}/{model_stage_to_load}"
    try:
        model = mlflow.tensorflow.load_model(model_uri)
    except mlflow.exceptions.MlflowException as e:
        print(f"🚨 ERROR: ไม่สามารถโหลดโมเดลได้จาก URI {model_uri}. โปรดตรวจสอบ Stage/Version: {e}")
        return # หยุดการทำงานหากโหลดโมเดลไม่ได้
        
    # --- โหลดข้อมูลและประเมินผล ---
    print("📂 โหลดข้อมูล Validation/Test Set...")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH, validation_split=0.2, subset="validation", seed=42, 
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    class_names = test_ds.class_names
    y_true, y_pred = [], []
    
    print("🧠 ประเมินโมเดล...")
    # ... (ส่วนการประเมินผลเหมือนเดิม) ...
    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"✅ Test Accuracy: {acc:.4f}")

    # 3. 💡 Log Metrics & Artifacts
    with mlflow.start_run(run_name=f"evaluation_for_{model_stage_to_load}") as run:
        mlflow.log_metric("test_accuracy", acc)
        # ... (สร้างและบันทึก Confusion Matrix เหมือนเดิม) ...
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(xticks_rotation=45)
        os.makedirs("evaluation_artifacts", exist_ok=True)
        cm_path = "evaluation_artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        print(f"📊 ประเมินโมเดลเสร็จสิ้น และบันทึกผลใน MLflow Run ID: {run.info.run_id}")

    # 4. 💡 ขั้นตอนการย้าย Stage (Transition Logic)
    # เราจะย้าย Stage ไป 'Staging' เฉพาะเมื่อมีการส่ง Argument 'Latest' เข้ามา
    if model_stage_to_load.lower() == 'latest' and acc >= THRESHOLD:
        try:
            client = mlflow.tracking.MlflowClient()
            # ค้นหา Version ที่เป็น 'Latest'
            latest_version = client.get_latest_versions(model_name_to_load, stages=['None'])[0].version
            
            client.transition_model_version_stage(
                name=model_name_to_load,
                version=latest_version,
                stage="Staging"
            )
            print(f"🚀 โมเดล {model_name_to_load} Version {latest_version} ถูกย้ายไป Stage 'Staging' แล้ว!")
            return True
        except Exception as e:
            print(f"🚨 ERROR: ไม่สามารถย้าย Stage โมเดลได้: {e}")
            return False
    elif model_stage_to_load.lower() == 'latest':
        print(f"⚠️ Accuracy {acc:.4f} ต่ำกว่าเกณฑ์ {THRESHOLD} ไม่ย้าย Stage โมเดล")

    return True # คืนค่า True หากเป็นแค่การประเมินผล

if __name__ == "__main__":
    # 💡 เปลี่ยนชื่อฟังก์ชันที่เรียก
    evaluate_and_transition_model()