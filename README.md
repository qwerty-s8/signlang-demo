Sign Language Classification — Browser UI + FastAPI ONNX
Real‑time Sign Language classification with a lightweight browser UI and a FastAPI backend powered by ONNX Runtime. The browser captures 160×160 JPEG snapshots and sends them to /predict; the server returns label and confidence.

Live demo
Frontend: [<https://<your-pages-site>/>](https://qwerty-s8.github.io/signlang-demo/)

API base: https://signlang-demo.onrender.com

Mode switches

Default: API mode (stable on all devices)

Force Local ONNX (optional): add ?mode=local after enabling the ONNX script tag in index.html

Features
Real‑time webcam capture, selfie‑view mirroring.

Server‑side ONNX inference with temporal smoothing in UI.

Live stats: Prediction, Confidence, EMA‑FPS; session summary on Stop.

Privacy: only downscaled JPEG snapshots are sent; nothing is stored.

Repo layout
index.html — UI markup (ONNX tag commented by default)

style.css — theme

app_web.js — browser client (API mode; optional local mode switch)

backend/

app_fastapi.py — FastAPI inference service

weights/best.onnx — model weights (server side)

labels.txt — model label order

requirements.txt — Python deps

How it works
Browser: getUserMedia → mirror → resize to 160×160 → JPEG → POST /predict every ~300 ms.

Server: ONNX Runtime runs best.onnx and returns JSON: { ok, label, conf }.

Client UI: color cues for low confidence (< 0.70), smoothing across last N frames.

Quick start
Frontend (local)

Open index.html via a static server (VS Code Live Server).

Use HTTPS for camera on the web. Hard refresh after changes (Ctrl+F5).

Backend (local)

python -m venv .venv && .venv/Scripts/activate (Windows)

pip install -r backend/requirements.txt

uvicorn backend.app_fastapi:app --host 0.0.0.0 --port 8787

Set API_URL in app_web.js to http://localhost:8787/predict for local testing.

Deploy backend (Render/any cloud)

Place best.onnx at backend/weights/best.onnx or set MODEL_PATH.

Ensure /health returns { "ok": true }.

Configure CORS for your Pages origin or allow all for demo.

API
Endpoint

POST /predict

Request (multipart/form-data)

file: JPEG image (160×160 recommended)

Response (JSON)

{ "ok": true, "label": "A", "conf": 0.92 }

Example

curl -F "file=@test.jpg" https://signlang-demo.onrender.com/predict

Note

Client maps flexible keys: label|pred and conf|score. Keep labels.txt in sync with training.

Configuration
Client

API_URL: backend /predict endpoint

SEND_SIZE: 160 (input size)

SEND_MS: 300 (send cadence)

WARN_THRESH: 0.70 (UI cue)

Server (typical)

MODEL_PATH=backend/weights/best.onnx

LABELS=backend/labels.txt

Model
Input: RGB, normalized, resized to 160×160 (or 224×224 as trained).​

Labels: digits 1–9 and A–Z (custom order).

In API mode, inference is always on your server model; the browser doesn’t load weights.

Accuracy
Validation top‑1: ~92–95% (example; update with your measured results).

Real‑time varies by lighting and background; smoothing stabilizes outputs.

Add confusion matrix and macro‑F1 when you log evaluation.

Troubleshooting
Camera not opening:

Use HTTPS, allow permission, ensure only one script includes app_web.js.

Frames = 0, no predictions:

DevTools → Network should show /predict 200s every ~300 ms.

404/405: wrong route or method; 415/422: send multipart “file”.

Fix CORS on server if blocked.

“Aw, Snap! Out of Memory”:

Keep ONNX script commented; default to API mode.
