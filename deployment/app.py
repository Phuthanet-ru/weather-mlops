from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import mlflow
import mlflow.tensorflow # เพิ่มมาเพื่อใช้ load_model
import io
import os

app = FastAPI(title="🌦️ Weather Classifier API", version="1.0")

# ✅ Path ที่ถูกต้องคือ โฟลเดอร์รากของ Artifacts ที่ถูกคัดลอกมา
MODEL_PATH = "./model" 
print(f"📦 Loading model from local path: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    # ปรับปรุง Error message ให้ชัดเจนขึ้น
    raise FileNotFoundError(f"❌ Model path not found at: {MODEL_PATH}")

try:
    # 💡 ใช้ mlflow.tensorflow.load_model() เพื่อจัดการโครงสร้าง Artifacts
    # ถ้าโมเดลถูก Log ด้วย mlflow.tensorflow.log_model
    model = mlflow.tensorflow.load_model(MODEL_PATH) 
except Exception as e:
    # หาก load ด้วย MLflow ไม่ได้ ให้ลอง load ด้วย Keras ธรรมดา
    print(f"⚠️ Failed to load model with MLflow: {e}. Trying Keras load...")
    try:
        # หาก model.keras อยู่ใน ./model/data/
        model = tf.keras.models.load_model(f"{MODEL_PATH}/data") 
    except Exception as keras_e:
        raise RuntimeError(f"❌ Critical Error: Cannot load model via MLflow or Keras. Details: {keras_e}")

CLASS_NAMES = ["cloudy", "foggy", "rainy", "snowy", "sunny"]
IMG_SIZE = (128, 128)

@app.get("/")
def root():
    return {"message": "Weather Classifier API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ... (ส่วน predict เหมือนเดิม)
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        preds = model.predict(image_array)
        pred_class = CLASS_NAMES[np.argmax(preds)]
        confidence = float(np.max(preds))

        return JSONResponse({
            "predicted_class": pred_class,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)