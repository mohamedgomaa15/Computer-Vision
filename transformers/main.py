import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from contextlib import asynccontextmanager

onnx_model_path = "vit_model.onnx"

ort_session = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ort_session
    ort_session = ort.InferenceSession(onnx_model_path)
    yield
    ort_session = None

app = FastAPI(lifespan=lifespan)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesID(image_bytes).convert("RGB"))
    img = img.resize((224, 224))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)/255.
    return img

@app.post('/infer/')
async def infer(file:UploadFile=File(...)):
    if ort_session is None:
        return {"Error": "Model is Not loaded"}
    image_bytes = await file.read()
    image_processed = preprocess_image(image_bytes)
    session_input = {ort_session.get_inputs()[0].name: image_processed}
    onnx_outputs = ort_session.run(None, session_input)
    onnx_logits = onnx_outputs[0]
    pred_idx = int(np.argmax(onnx_logits)[0])
    return {"predicted_idx": pred_idx}

@app.get('/gg/')
def test_api():
    return {"Message": "API is live now !!!"}