// app_web.js — Browser-only ONNX inference (refined)

// ------------------ Configuration ------------------
const LABELS = "Z,Y,X,W,V,U,T,S,R,Q,P,O,N,M,L,K,J,I,H,G,F,E,1,D,C,B,A,9,8,7,6,5,4,3,2".split(',');
const SIZE = 224;
const MIRROR = true;           // selfie view
const INTERVAL_MS = 250;       // ~4 Hz
const SMOOTH = 15;             // temporal smoothing window
const WARN_THRESH = 0.70;      // low-confidence color cue

// ------------------ UI Elements --------------------
const statusEl = document.getElementById('status');
const video    = document.getElementById('video');
const runBtn   = document.getElementById('runBtn');
const stopBtn  = document.getElementById('stopBtn');
const predEl   = document.getElementById('pred');
const confEl   = document.getElementById('conf');
const fpsEl    = document.getElementById('fps');
const summaryEl= document.getElementById('summary'); // optional summary card

// ------------------ State --------------------------
let running = false;
let timer   = null;
let session = null;
let inputName = null;

// Offscreen canvas for preprocessing
const canvas = document.createElement('canvas');
canvas.width = SIZE; canvas.height = SIZE;
const ctx = canvas.getContext('2d', { willReadFrequently: true });

// FPS (EMA) + smoothing + session metrics
let fpsEMA = 0.0, tPrev = performance.now();
const smoothQ = [];
let frameCount = 0;
const labelHist = new Map();

function setStatus(txt, on=false) {
  statusEl.textContent = txt;
  statusEl.classList.toggle('on', on);
  statusEl.classList.toggle('off', !on);
}

function colorByConf(v) {
  const low = v < WARN_THRESH;
  const c = low ? '#f4bf4f' : '#e9ecf1';
  predEl.style.color = c;
  confEl.style.color = c;
}

function resetSessionMetrics() {
  fpsEMA = 0.0; tPrev = performance.now();
  smoothQ.length = 0;
  frameCount = 0;
  labelHist.clear();
  predEl.textContent = '—';
  confEl.textContent = '—';
  fpsEl.textContent  = '—';
  if (summaryEl) summaryEl.innerHTML = '<p class="muted">Running… a summary will appear here when you stop.</p>';
}

// ------------------ Camera & Model -----------------
async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await video.play();
}

async function initSession() {
  // Optional: set WASM path to explicit CDN
  // ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  session = await ort.InferenceSession.create("weights/best.onnx");
  inputName = session.inputNames[0];
}

// ------------------ Preprocess ---------------------
function preprocessToNCHW() {
  // Draw with mirror for selfie
  if (MIRROR) {
    ctx.save();
    ctx.translate(SIZE, 0); ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, SIZE, SIZE);
    ctx.restore();
  } else {
    ctx.drawImage(video, 0, 0, SIZE, SIZE);
  }

  // HWC RGBA -> NCHW RGB, normalized 0..1
  const data = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const H = SIZE, W = SIZE;
  const out = new Float32Array(1 * 3 * H * W);
  let p = 0, rIdx = 0, gIdx = H*W, bIdx = 2*H*W;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      out[rIdx + p] = data[i]   / 255;
      out[gIdx + p] = data[i+1] / 255;
      out[bIdx + p] = data[i+2] / 255;
      p++;
    }
  }
  return new ort.Tensor("float32", out, [1, 3, H, W]);
}

// ------------------ Math helpers -------------------
function softmax(logits) {
  let m = -Infinity;
  for (let v of logits) if (v > m) m = v;
  const exps = logits.map(v => Math.exp(v - m));
  const s = exps.reduce((a,b)=>a+b, 0);
  return exps.map(v => v / s);
}
function pushProb(vec) {
  smoothQ.push(vec);
  if (smoothQ.length > SMOOTH) smoothQ.shift();
}
function avgProb() {
  if (!smoothQ.length) return null;
  const m = smoothQ[0].length;
  const out = new Float32Array(m);
  for (let i=0;i<smoothQ.length;i++)
    for (let j=0;j<m;j++) out[j] += smoothQ[i][j];
  for (let j=0;j<m;j++) out[j] /= smoothQ.length;
  return out;
}
function trackLabel(lbl) { labelHist.set(lbl, (labelHist.get(lbl)||0) + 1); }

// ------------------ Inference step -----------------
async function step() {
  try {
    const tensor = preprocessToNCHW();
    const out = await session.run({ [inputName]: tensor });
    const outName = session.outputNames[0];
    const logits = Array.from(out[outName].data);
    const probs = (logits.every(v => v>=0 && v<=1) && Math.abs(logits.reduce((a,b)=>a+b,0)-1)<1e-3)
      ? logits : softmax(logits);

    pushProb(Float32Array.from(probs));
    const avg = avgProb() || Float32Array.from(probs);

    // Top-1
    let top = 0; for (let i=1;i<avg.length;i++) if (avg[i] > avg[top]) top = i;
    const label = LABELS[top] || '?';
    const conf  = avg[top];

    // UI
    predEl.textContent = label;
    confEl.textContent = conf.toFixed(2);
    colorByConf(conf);

    // FPS
    const tNow = performance.now();
    const dt = Math.max(1e-3, (tNow - tPrev)/1000);
    tPrev = tNow;
    fpsEMA = 0.9*fpsEMA + 0.1*(1.0/dt);
    fpsEl.textContent = fpsEMA.toFixed(1);

    // Metrics
    frameCount++;
    trackLabel(label);
  } catch (e) {
    // Non-fatal: show a transient status but keep loop running
    console.warn('step error:', e);
  }
}

// ------------------ Session summary ----------------
function renderSummary() {
  let total = 0, topLabel = '—', topCount = 0;
  for (const [,cnt] of labelHist) total += cnt;
  for (const [k,c] of labelHist) if (c > topCount) { topCount = c; topLabel = k; }

  const share = total ? ((topCount/total)*100).toFixed(1) + '%' : '—';
  const avgFps = fpsEl.textContent || '—';
  const lastConf = confEl.textContent || '—';

  const html = `
    <p><span class="badge ok">Session complete</span></p>
    <p><strong>Frames:</strong> ${frameCount}</p>
    <p><strong>Avg FPS (EMA):</strong> ${avgFps}</p>
    <p><strong>Top class:</strong> ${topLabel} <span class="muted">(${share} of frames)</span></p>
    <p><strong>Last confidence:</strong> ${lastConf}</p>
  `;
  if (summaryEl) summaryEl.innerHTML = html;

  console.log(`[SUMMARY] frames=${frameCount} avg_fps=${avgFps} top=${topLabel} share=${share} last_conf=${lastConf}`);
}

// ------------------ Controls -----------------------
async function start() {
  if (running) return;
  setStatus('Starting…', true);
  try {
    if (!video.srcObject) await initCamera();
    if (!session) await initSession();
  } catch (e) {
    setStatus('Camera/Model error', false);
    console.error(e);
    return;
  }

  resetSessionMetrics();
  running = true;
  setStatus('Running', true);
  timer = setInterval(step, INTERVAL_MS);
}

function stop() {
  if (!running) return;
  running = false;
  if (timer) { clearInterval(timer); timer = null; }
  setStatus('Idle', false);
  renderSummary();
}

// Wire up
runBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
setStatus('Idle', false);
