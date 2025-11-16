// app_web.js
// Configuration
const LABELS = "Z,Y,X,W,V,U,T,S,R,Q,P,O,N,M,L,K,J,I,H,G,F,E,1,D,C,B,A,9,8,7,6,5,4,3,2".split(',');
const SIZE = 224;
const MIRROR = true;          // selfie view
const INTERVAL_MS = 250;      // ~4 Hz
const SMOOTH = 15;            // temporal smoothing (frames)

// UI
const statusEl = document.getElementById('status');
const video = document.getElementById('video');
const runBtn = document.getElementById('runBtn');
const stopBtn = document.getElementById('stopBtn');
const predEl = document.getElementById('pred');
const confEl = document.getElementById('conf');
const fpsEl  = document.getElementById('fps');

let running = false;
let timer = null;
let session = null;
let inputName = null;

// Offscreen canvas for preprocessing
const canvas = document.createElement('canvas');
canvas.width = SIZE; canvas.height = SIZE;
const ctx = canvas.getContext('2d', { willReadFrequently: true });

// Simple EMA FPS
let fps = 0.0, tLast = performance.now();

// Smoothing queue of probability vectors
const q = [];
function pushProb(vec) {
  q.push(vec);
  if (q.length > SMOOTH) q.shift();
}
function avgProb() {
  if (q.length === 0) return null;
  const n = q.length, m = q[0].length;
  const out = new Float32Array(m);
  for (let i=0;i<n;i++) for (let j=0;j<m;j++) out[j] += q[i][j];
  for (let j=0;j<m;j++) out[j] /= n;
  return out;
}

function setStatus(txt, on=false) {
  statusEl.textContent = txt;
  statusEl.classList.toggle('on', on);
  statusEl.classList.toggle('off', !on);
}

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await video.play();
}

async function initSession() {
  // Optional: ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  session = await ort.InferenceSession.create("weights/best.onnx");
  inputName = session.inputNames[0];
}

function preprocessToNCHW() {
  // Draw frame with optional mirror, square fill to 224x224
  if (MIRROR) {
    ctx.save();
    ctx.translate(SIZE, 0); ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, SIZE, SIZE);
    ctx.restore();
  } else {
    ctx.drawImage(video, 0, 0, SIZE, SIZE);
  }

  const data = ctx.getImageData(0, 0, SIZE, SIZE).data; // RGBA
  const C = 3, H = SIZE, W = SIZE;
  const out = new Float32Array(1 * C * H * W);
  let p = 0, rIdx = 0, gIdx = H*W, bIdx = 2*H*W;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const r = data[i] / 255, g = data[i+1] / 255, b = data[i+2] / 255;
      out[rIdx + p] = r;
      out[gIdx + p] = g;
      out[bIdx + p] = b;
      p++;
    }
  }
  return new ort.Tensor("float32", out, [1, 3, H, W]);
}

function softmax(logits) {
  let max = -Infinity;
  for (let v of logits) if (v > max) max = v;
  const exps = logits.map(v => Math.exp(v - max));
  const sum = exps.reduce((a,b)=>a+b,0);
  return exps.map(v => v/sum);
}

async function step() {
  const t0 = performance.now();
  const tensor = preprocessToNCHW();
  const out = await session.run({ [inputName]: tensor });
  const outName = session.outputNames[0];
  const logits = Array.from(out[outName].data);
  const probs = (logits.every(v => v>=0 && v<=1) && Math.abs(logits.reduce((a,b)=>a+b,0)-1)<1e-3)
    ? logits
    : softmax(logits);

  pushProb(Float32Array.from(probs));
  const avg = avgProb() || Float32Array.from(probs);
  let top = 0; for (let i=1;i<avg.length;i++) if (avg[i] > avg[top]) top = i;

  const t1 = performance.now();
  const dt = Math.max(1e-3, (t1 - tLast)/1000);
  tLast = t1;
  fps = 0.9*fps + 0.1*(1.0/dt);

  predEl.textContent = LABELS[top] || '?';
  confEl.textContent = avg[top].toFixed(2);
  fpsEl.textContent = fps.toFixed(1);
}

async function start() {
  if (running) return;
  setStatus('Starting…', true);
  if (!video.srcObject) await initCamera();
  if (!session) await initSession();

  running = true;
  setStatus('Running', true);
  timer = setInterval(step, INTERVAL_MS);
}

function stop() {
  if (!running) return;
  running = false;
  setStatus('Idle', false);
  if (timer) { clearInterval(timer); timer = null; }
  predEl.textContent = '—';
  confEl.textContent = '—';
  fpsEl.textContent  = '—';
}

runBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
setStatus('Idle', false);
