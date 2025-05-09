from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
import uvicorn

# Load model function
def load_model():
    model = tf.keras.models.load_model("model/xray_model.hdf5", compile=False)
    return model

model = load_model()

# FastAPI App initialization
app = FastAPI()

# Define the response model
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float

# Define the image processing and prediction function
def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Define the class names (labels)
class_names = ['NORMAL', 'PNEUMONIA']

# Define the API route for prediction
@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))

    # Get prediction
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])

    # Prepare the response
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Return the result as a JSON response
    return PredictionResponse(predicted_class=predicted_class, confidence=confidence)

# Run the app if the script is called directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# if __name__ == "__main__":
#     import os
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port, debug=True)