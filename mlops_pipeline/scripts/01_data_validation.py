import os
from PIL import Image
import mlflow

def validate_data(data_dir="data"):
    """
    ตรวจสอบความถูกต้องของชุดข้อมูลภาพ (5 คลาส)
    และบันทึกผลลงใน MLflow
    """
    mlflow.set_experiment("Weather Classification - Data Validation")

    with mlflow.start_run():
        mlflow.set_tag("ml.step", "data_validation")
        print(f"🔍 เริ่มตรวจสอบข้อมูลใน: {data_dir}")

        class_counts = {}
        total_images = 0
        corrupted_images = 0

        # ตรวจทุกโฟลเดอร์ (cloudy, foggy, rainy, snowy, sunny)
        for cls in os.listdir(data_dir):
            cls_path = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            files = os.listdir(cls_path)
            class_counts[cls] = len(files)
            total_images += len(files)

            for f in files:
                img_path = os.path.join(cls_path, f)
                try:
                    img = Image.open(img_path)
                    img.verify()  # ตรวจสอบว่าเปิดได้จริง
                except Exception:
                    corrupted_images += 1

        # log ผลลัพธ์
        mlflow.log_metric("total_images", total_images)
        mlflow.log_metric("corrupted_images", corrupted_images)
        mlflow.log_param("num_classes", len(class_counts))
        mlflow.log_dict(class_counts, "class_distribution.json")

        print(f"🟩 ตรวจสอบข้อมูลสำเร็จ: {len(class_counts)} classes, {total_images} images")
        print(f"🟥 พบไฟล์เสีย: {corrupted_images}")
        print(f"📊 การกระจายข้อมูล: {class_counts}")

if __name__ == "__main__":
    validate_data("mlops_pipeline/data")
