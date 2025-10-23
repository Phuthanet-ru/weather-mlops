from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import mlflow
import tensorflow as tf
import io

app = FastAPI(title="🌦️ Weather Classifier API", version="1.0")

# โหลดโมเดลจาก MLflow Model Registry
mlflow.set_tracking_uri("file:./mlruns")
MODEL_NAME = "weather-classifier-prod"
STAGE = "Staging"

print("📦 Loading model from MLflow Registry...")
model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/{STAGE}")

CLASS_NAMES = ["cloudy", "foggy", "rainy", "snowy", "sunny"]
IMG_SIZE = (128, 128)

@app.get("/")
def root():
    return {"message": "Weather Classifier API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # อ่านและแปลงรูปภาพ
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        # ทำนาย
        preds = model.predict(image_array)
        pred_class = CLASS_NAMES[np.argmax(preds)]
        confidence = float(np.max(preds))

        return JSONResponse({
            "predicted_class": pred_class,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
