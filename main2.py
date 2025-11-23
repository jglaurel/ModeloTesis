from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from keras.utils import img_to_array
from keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Cargar el modelo entrenado
model = load_model("resnet50_v21.h5", compile=False)

# Tamaño esperado por ResNet50
IMG_SIZE = (224, 224)

# Diccionario de clases (ajusta según tu dataset)
class_labels = {0: "NORMAL", 1: "PNEUMONIA"}

@app.get("/")
def home():
    return {"mensaje": "API para predecir neumonía con ResNet50"}

@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # 👈 mismo preprocesamiento que en entrenamiento
    
    pred = model.predict(img_array)[0][0]  # salida sigmoide
    
    # Umbral 0.5
    clase = 1 if pred >= 0.5 else 0
    
    return {
        "clase_predicha": class_labels[clase],
        "confianza": float(pred if clase == 1 else 1 - pred)
    }