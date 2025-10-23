# -----------------------------
# 🌤️ Weather Classifier Dockerfile
# -----------------------------
# ---- Base Image ----
    FROM python:3.10-slim

    # ---- Working Directory ----
    WORKDIR /app
    
    # ---- Copy requirements and install dependencies ----
    # 💡 คัดลอก requirements จาก mlops_pipeline และติดตั้ง
    COPY mlops_pipeline/requirements.txt .
    RUN pip install --no-cache-dir --upgrade pip && \
        pip install -r requirements.txt && \
        # 💡 เพิ่ม MLflow, uvicorn และ Pillow
        pip install fastapi uvicorn pillow mlflow tensorflow
    
    # ---- Copy API code and Model Artifacts ----
    # คัดลอกโฟลเดอร์ 'deployment' ทั้งหมด
    COPY deployment /app/deployment 
    
    # ✅ คัดลอกโฟลเดอร์ 'model' ซึ่งมี Artifacts เข้าไปใน /app/model
    COPY model/ /app/model/ 
    
    # ---- Expose port ----
    EXPOSE 8000
    
    # ---- Run FastAPI ----
    # 💡 app.py ถูกย้ายไปที่ /app/deployment/app.py ดังนั้น module path คือ deployment.app
    CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]