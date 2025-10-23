from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os

app = FastAPI(title="🌦️ Weather Classifier API", version="1.0")

# ✅ โหลดโมเดลจาก local path (ไม่ต้องพึ่ง MLflow registry)
MODEL_PATH = "model/weather_cnn_model"
print(f"📦 Loading model from local path: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model path not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["cloudy", "foggy", "rainy", "snowy", "sunny"]
IMG_SIZE = (128, 128)

@app.get("/")
def root():
    return {"message": "Weather Classifier API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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

