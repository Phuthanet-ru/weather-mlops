from pathlib import Path
from PIL import Image
import mlflow
# F401 fix: ลบ import os และ import json ที่ไม่ได้ใช้
# เนื่องจาก Path และ mlflow.log_dict ถูกใช้แทน
mlflow.set_tracking_uri("file:./mlruns")


def validate_data(data_dir="mlops_pipeline/data"):
    """
    ตรวจสอบความถูกต้องของชุดข้อมูลภาพ (5 คลาส)
    และบันทึกผลลงใน MLflow
    """
    # W291 fix: ลบ Trailing Whitespace ใน Docstring
    data_path = Path(data_dir)

    # W293 fix: ลบ whitespace ในบรรทัดว่าง

    # 💡 ใช้ชื่อ Experiment ที่ชัดเจน
    mlflow.set_experiment("Weather Classification - Data Validation")

    with mlflow.start_run():
        mlflow.set_tag("ml.step", "data_validation")
        print(f"🔍 เริ่มตรวจสอบข้อมูลใน: {data_path.resolve()}")

        class_counts = {}
        total_images = 0
        corrupted_images = 0

        # W293 fix: ลบ whitespace ในบรรทัดว่าง

        # 💡 เก็บข้อมูล metadata สำหรับ log artifact
        validation_metadata = {
            "root_path": str(data_path.resolve()),
            "classes_checked": [],
            "image_counts": {},
            "corrupted_log": []
        }

        # ตรวจทุกโฟลเดอร์ (cloudy, foggy, rainy, snowy, sunny)
        for cls_dir in data_path.iterdir():
            if not cls_dir.is_dir():
                continue

            cls = cls_dir.name
            validation_metadata["classes_checked"].append(cls)

            files = list(cls_dir.glob("*"))

            # 💡 นับจำนวนไฟล์ในคลาสนี้
            current_class_count = 0

            for f in files:
                if f.is_file():
                    total_images += 1
                    current_class_count += 1

                    try:
                        img = Image.open(f)
                        img.verify()  # ตรวจสอบว่าเปิดได้จริง
                    except Exception:
                        corrupted_images += 1
                        # E501 fix: ตัดบรรทัด
                        validation_metadata["corrupted_log"].append(
                            str(f.resolve()))

            class_counts[cls] = current_class_count
            validation_metadata["image_counts"][cls] = current_class_count

        # -------------------- MLflow Logging --------------------

        # W293 fix: ลบ whitespace ในบรรทัดว่าง

        # Log Metrics
        mlflow.log_metric("total_images", total_images)
        mlflow.log_metric("corrupted_images", corrupted_images)
        mlflow.log_param("num_classes", len(class_counts))

        # Log Artifacts (Metadata)
        # 💡 log_dict() ใช้ได้ดีกับ JSON
        mlflow.log_dict(class_counts, "artifacts/class_distribution.json")
        # E501 fix: ตัดบรรทัด + W291 fix
        mlflow.log_dict(validation_metadata,
                        "artifacts/validation_metadata.json")

        # -------------------- Console Output --------------------
        # E501 fix: ตัดบรรทัด
        print(f"🟩 ตรวจสอบข้อมูลสำเร็จ: {len(class_counts)} classes, "
              f"{total_images} images")
        print(f"🟥 พบไฟล์เสีย: {corrupted_images}")
        print(f"📊 การกระจายข้อมูล: {class_counts}")


if __name__ == "__main__":
    validate_data("mlops_pipeline/data")
# W292 fix: เพิ่มบรรทัดว่างเปล่าที่ท้ายไฟล์
