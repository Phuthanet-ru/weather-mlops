import os
import tensorflow as tf

# ❗ แก้ไข Path นี้ให้ตรงกับโฟลเดอร์ data ของคุณ
DATA_ROOT = "mlops_pipeline/data" 
if not os.path.isabs(DATA_ROOT):
    DATA_ROOT = os.path.join(os.getcwd(), DATA_ROOT)

ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp') 

print("--- เริ่มตรวจสอบไฟล์ภาพที่ TensorFlow อ่านไม่ได้ ---")
corrupted_found = False

for root, _, files in os.walk(DATA_ROOT):
    for file in files:
        file_path = os.path.join(root, file)
        
        # ตรวจสอบเฉพาะไฟล์ที่มีนามสกุลที่อนุญาต
        if file.lower().endswith(ALLOWED_EXTENSIONS):
            try:
                # 1. อ่านไฟล์เป็นไบนารี
                img_bytes = tf.io.read_file(file_path)
                # 2. พยายามถอดรหัส (Decode) ซึ่งเป็นขั้นตอนที่ Keras ติด
                # ใช้ try/except เพื่อจับข้อผิดพลาด 'Unknown image file format'
                _ = tf.image.decode_image(img_bytes, channels=3)
            
            except tf.errors.InvalidArgumentError as e:
                if "Unknown image file format" in str(e):
                    print(f"🚨 ไฟล์เสียหายที่พบ (TF ไม่รองรับ): {file_path}")
                    corrupted_found = True
            except Exception as e:
                # จับข้อผิดพลาดอื่น ๆ ที่อาจเกิดขึ้น
                pass 

print("--- ตรวจสอบเสร็จสิ้น ---")
if not corrupted_found:
    print("✅ ไม่พบไฟล์ภาพที่เสียหายที่ TensorFlow อ่านไม่ได้")
else:
    print("⚠️ โปรดลบหรือย้ายไฟล์ที่ระบุออก แล้วลองรัน 03_train_model.py อีกครั้ง")