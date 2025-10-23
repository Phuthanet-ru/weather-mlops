# -----------------------------
# 🌤️ Weather Classifier Dockerfile
# -----------------------------
# ---- Base Image ----
    FROM python:3.10-slim

    # ---- Working Directory ----
    WORKDIR /app
    
    # ---- Copy requirements and install dependencies ----
    COPY mlops_pipeline/requirements.txt .
    RUN pip install --no-cache-dir --upgrade pip && \
        pip install -r requirements.txt && \
        # 🚨 แก้ไข: เพิ่ม python-multipart เข้าไป
        pip install fastapi uvicorn pillow mlflow tensorflow python-multipart
    
    # ---- Copy API code and Model Artifacts ----
    COPY deployment /app/deployment 
    COPY model/ /app/model/ 
    
    # ---- Expose port ----
    EXPOSE 8000
    
    # ---- Run FastAPI ----
    CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]