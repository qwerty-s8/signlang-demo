Sign Language Classification Demo (ONNX + Python)
A local, privacy‑first demo that recognizes hand signs from webcam video using an ONNX classification model. The project provides a clean web UI, a lightweight Python server, and a real‑time ONNX Runtime pipeline for smooth, low‑latency inference.

Demo highlights
Real‑time webcam inference with temporal smoothing for stable predictions.

Clean website to start/stop the demo and view live Pred/Conf/FPS.

Simple local setup on Windows; no cloud required.

Table of contents
Features

Tech stack

Project structure

Quick start (Windows)

Usage

Configuration

Troubleshooting

Dataset and class order

Performance tips

Roadmap

Contributing

License

Acknowledgments

Features
ONNX model inference with ONNX Runtime (CPU).

Webcam capture, preprocessing (crop/pad, mirror, normalization).

Temporal probability averaging and confidence thresholding for stability.

Website shows live label, confidence, and FPS via a /stats endpoint.

Tech stack
Python: onnxruntime, opencv-python, numpy

Frontend: HTML/CSS/JavaScript (vanilla)

Local server: Python http.server subclass (custom endpoints)

Project structure
text
SignLangDemoSite/
├─ index.html
├─ style.css
├─ app.js
├─ server.py
├─ webcam_onnx_cls.py
├─ weights/
│  ├─ best.onnx
│  └─ (optional *.pt for reference, not used at runtime)
├─ assets/
│  ├─ screenshot_demo.png
│  └─ confusion_matrix.png
├─ latest_stats.json         # written by the app at runtime
├─ start_demo.bat / .vbs     # optional launch helpers
├─ launch/                   # optional helpers
└─ .venv/                    # local virtual env (not committed)
Quick start (Windows)
Create and activate a virtual environment:

py -m venv .venv

..venv\Scripts\Activate.ps1

Install dependencies:

pip install onnxruntime opencv-python numpy

Start the local servers:

Terminal A: python -u server.py

Terminal B: python -m http.server 8080

Open the website:

http://127.0.0.1:8080/index.html

Click Run Demo. A camera window opens; press q to quit.

Usage
Run Demo: Launches the ONNX app in a separate process and starts publishing live stats (Pred/Conf/FPS) to /stats.

Stop Demo: Brings UI back to Idle; close the camera window with q.

Live stats: UI shows top label, average confidence, and FPS. Browser console logs [STAT] lines.

Configuration
Common runtime flags passed to the Python script:

--model .\weights\best.onnx

--imgsz 224

--mirror (flip horizontally for selfie view)

--smooth 15 (temporal window)

--conf_thresh 0.70

--pad_square (use padding instead of center crop)

--imagenet_norm (enable only if training used mean/std)

--labels "<your EXACT class order>"

Example custom label order (matches this dataset):

"Z,Y,X,W,V,U,T,S,R,Q,P,O,N,M,L,K,J,I,H,G,F,E,1,D,C,B,A,9,8,7,6,5,4,3,2"

Note: The label sequence must match the training class index order, or predictions will be mapped to wrong names.

Troubleshooting
Button returns 200 but nothing happens: ensure server.py and http.server are both running; check that only one process listens on port 8787; kill stale processes.

/stats returns 501: update server.py so do_GET (inside the handler class) implements the /stats branch, and restart the correct server instance.

Wrong predictions: verify label order, turn on --mirror, increase --smooth and --conf_thresh, match preprocessing to training (crop vs pad, RGB, normalization).

No camera feed: confirm permissions and only one app accessing the webcam.

Dataset and class order
This project trained with folder‑based classification. Training class order follows the alphabetical sort of class folders at training time.

Confirm order by listing the train subfolders and copy that exact sequence into --labels.

For this repo, the used order is:
"Z,Y,X,W,V,U,T,S,R,Q,P,O,N,M,L,K,J,I,H,G,F,E,1,D,C,B,A,9,8,7,6,5,4,3,2"

Performance tips
Ensure the hand occupies 40–60% of the frame; use front lighting; keep stable distance.

Use temporal averaging and a higher threshold to reduce flicker.

CPU is sufficient for 224×224; try a smaller imgsz for more FPS on low‑end CPUs.

Roadmap
Optional browser‑only demo using onnxruntime‑web (WASM), hosted on GitHub Pages.

Optional FastAPI inference API for remote deployment (container/VM).

Add top‑3 label overlay and confusion logging.

Contributing
Pull requests and issues are welcome.

Open an issue describing a bug or feature.

For contributions, fork the repo, create a feature branch, and open a PR with a clear description and test plan.

License
Include your chosen license here (e.g., MIT). Add a LICENSE file in the repo root.

Acknowledgments
ONNX Runtime for fast cross‑platform inference.

Ultralytics YOLOv8 classification pipeline used for model training.

How to include this in your repo

Create README.md in the project root.

Paste this content, then adjust screenshots, the exact labels string, and any custom flags.

Commit and push:

git add README.md

git commit -m "Add professional README"

git push

Optional extras for a polished repo

Badges (Python version, license, stars).

A small GIF or mp4 showing the live demo.

requirements.txt and a minimal .gitignore:

.venv/

pycache/

runs/

latest_stats.json

weights/*.pt (keep .onnx if size permits)

If best.onnx is large, attach it to GitHub Releases and download on first run.
