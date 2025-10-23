# -----------------------------
# 🌤️ Weather Classifier Dockerfile
# -----------------------------
    FROM python:3.10-slim

    # ตั้ง working directory
    WORKDIR /app
    
    # คัดลอกไฟล์ requirements เข้ามา
    COPY mlops_pipeline/requirements.txt ./requirements.txt
    
    # ติดตั้ง dependencies ที่จำเป็น
    RUN pip install --no-cache-dir -r requirements.txt \
        && pip install fastapi uvicorn python-multipart pillow mlflow tensorflow
    
    # คัดลอกทุกไฟล์ในโปรเจกต์เข้ามาใน container
    COPY . .
    
    # เปิดพอร์ต 8000 (สำหรับ API)
    EXPOSE 8000
    
    # รันเซิร์ฟเวอร์ FastAPI
    CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
    
