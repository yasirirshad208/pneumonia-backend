from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
import uvicorn

def load_model():
    model = tf.keras.models.load_model("model/xray_model.hdf5", compile=False)
    return model

model = load_model()

app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with frontend origin for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float

def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

class_names = ['NORMAL', 'PNEUMONIA']

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return PredictionResponse(predicted_class=predicted_class, confidence=confidence)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
