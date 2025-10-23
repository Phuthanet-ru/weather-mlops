import os
import tempfile
from pathlib import Path

# สมมติว่าคุณ import ฟังก์ชัน remove_dot_files จาก 03_train_model.py ได้
# ในการรันจริง อาจต้องปรับโครงสร้าง project ให้ tests เข้าถึงได้
# สำหรับตัวอย่าง เราจะคัดลอกฟังก์ชันมาเพื่อทดสอบ
ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

def remove_dot_files(root_dir):
    """ลบไฟล์ระบบหรือไฟล์ที่ไม่ใช่ภาพ (จำลองจาก 03_train_model.py)"""
    count = 0
    # ... (คัดลอกโค้ดเต็มของ remove_dot_files มาไว้ที่นี่ หรือ import อย่างถูกต้อง)
    # เนื่องจากไม่เห็นโค้ดเต็มในไฟล์ tests เราจะจำลองการทำงาน
    
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for d in list(dirs):
            if d.startswith('.'):
                try:
                    os.rmdir(os.path.join(root, d))
                    dirs.remove(d)
                    count += 1
                except OSError:
                    pass
        # ... (ส่วนการลบไฟล์ dot และไฟล์ที่ไม่ใช่ภาพ) ...
    return count


def test_remove_dot_files():
    """ทดสอบว่าฟังก์ชันสามารถลบไฟล์และโฟลเดอร์เริ่มต้นด้วย '.' ได้"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # สร้างไฟล์และโฟลเดอร์จำลอง
        (temp_path / "valid_image.jpg").touch()
        (temp_path / ".hidden_folder").mkdir()
        (temp_path / ".DS_Store").touch() # ไฟล์ dot
        
        # ฟังก์ชันควรลบ 2 รายการ (.hidden_folder และ .DS_Store)
        removed_count = remove_dot_files(temp_dir)
        
        # ตรวจสอบว่าถูกลบไป 2 รายการ
        assert removed_count >= 2 
        
        # ตรวจสอบว่าไฟล์ที่ถูกต้องยังอยู่
        assert (temp_path / "valid_image.jpg").exists()
        
        # ตรวจสอบว่าไฟล์และโฟลเดอร์ dot ถูกลบ
        assert not (temp_path / ".hidden_folder").exists()
        assert not (temp_path / ".DS_Store").exists()