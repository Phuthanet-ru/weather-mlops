import os
import tensorflow as tf

# 🔧 โฟลเดอร์เก็บข้อมูล
DATA_ROOT = "mlops_pipeline/data"
if not os.path.isabs(DATA_ROOT):
    DATA_ROOT = os.path.join(os.getcwd(), DATA_ROOT)

ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

print("🔍 เริ่มตรวจสอบและลบไฟล์ภาพที่ TensorFlow อ่านไม่ได้...")
corrupted_files = []

for root, _, files in os.walk(DATA_ROOT):
    for file in files:
        file_path = os.path.join(root, file)

        # ตรวจสอบเฉพาะไฟล์ที่มีนามสกุลที่อนุญาต
        if file.lower().endswith(ALLOWED_EXTENSIONS):
            try:
                # พยายามอ่านและ decode เพื่อทดสอบว่า TensorFlow อ่านได้ไหม
                img_bytes = tf.io.read_file(file_path)
                _ = tf.image.decode_image(img_bytes, channels=3)
            except tf.errors.InvalidArgumentError as e:
                if "Unknown image file format" in str(e):
                    corrupted_files.append(file_path)
            except Exception:
                pass

# 🔧 ลบไฟล์ที่เสียออกทั้งหมด
if corrupted_files:
    print(f"⚠️ พบไฟล์เสีย {len(corrupted_files)} ไฟล์ จะทำการลบออก...")
    for f in corrupted_files:
        try:
            os.remove(f)
            print(f"🗑️ ลบแล้ว: {f}")
        except Exception as e:
            print(f"❌ ลบไฟล์ไม่สำเร็จ: {f} ({e})")
else:
    print("✅ ไม่พบไฟล์ที่เสียหาย")

# E501 fix: ตัดบรรทัดให้สั้นลง
print(f"📊 สรุปผล: ตรวจสอบ {len(corrupted_files)} ไฟล์ที่เสีย "
      f"และลบเรียบร้อยแล้ว")
# W292 fix: เพิ่มบรรทัดว่างเปล่าที่ท้ายไฟล์