# -----------------------------
# 🌤️ Weather Classifier Dockerfile
# -----------------------------
# ---- Base Image ----
    FROM python:3.10-slim

    # ---- Working Directory ----
    WORKDIR /app
    
    # ---- Copy project ----
    COPY mlops_pipeline/requirements.txt .
    
    # ---- Install dependencies ----
    RUN pip install --no-cache-dir --upgrade pip && \
        pip install -r requirements.txt && \
        pip install fastapi uvicorn pillow tensorflow
    
    # ---- Copy API code ----
    COPY deployment/app.py deployment/app.py
    
    # ---- Copy trained model ----
    # (ถูก log ไว้ที่ mlruns/... เราจะ copy มาจาก path ของ run ล่าสุดใน workflow)
    COPY model/ model/
    
    # ---- Expose port ----
    EXPOSE 8000
    
    # ---- Run FastAPI ----
    CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
    
