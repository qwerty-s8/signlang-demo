import io, base64, os
from typing import Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.environ.get("MODEL_PATH", "weights/best.onnx")
IMG_SIZE   = int(os.environ.get("IMG_SIZE", "224"))
LABELS     = os.environ.get("LABELS", "Z,Y,X,W,V,U,T,S,R,Q,P,O,N,M,L,K,J,I,H,G,F,E,1,D,C,B,A,9,8,7,6,5,4,3,2").split(',')
MIRROR     = os.environ.get("MIRROR", "1") == "1"

app = FastAPI(title="SignLang ONNX API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load once
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x, dtype=np.float32)
    return e / e.sum()

def preprocess(img: Image.Image, size=224, mirror=True) -> np.ndarray:
    if mirror: img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # square center-crop
    w, h = img.size
    side = min(w, h)
    l = (w - side) // 2; t = (h - side) // 2
    img = img.crop((l, t, l+side, t+side)).resize((size, size))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
    return arr

def infer(arr: np.ndarray):
    logits = sess.run([out_name], {in_name: arr})[0].squeeze()
    if logits.ndim == 0: logits = np.array([float(logits)], dtype=np.float32)
    probs = logits if (0 <= logits.min() and logits.max() <= 1 and abs(float(logits.sum())-1)<1e-3) else softmax(logits)
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])

@app.get("/health")
def health():
    return {"status": "ok"}

class B64Image(BaseModel):
    image: str  # data URL or pure base64

@app.post("/predict")
async def predict(file: Optional[UploadFile] = File(None), payload: Optional[B64Image] = None):
    try:
        if file is not None:
            img = Image.open(io.BytesIO(await file.read()))
        elif payload is not None and payload.image:
            b64 = payload.image.split(",")[-1]
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
        else:
            return {"ok": False, "error": "No image provided"}
        arr = preprocess(img, size=IMG_SIZE, mirror=MIRROR)
        idx, p = infer(arr)
        label = LABELS[idx] if 0 <= idx < len(LABELS) else "?"
        return {"ok": True, "label": label, "conf": round(float(p), 3)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
