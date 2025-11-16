// app_web.js — Browser-only ONNX inference (optimized + polished)

// ------------------ Configuration ------------------
const LABELS = "Z,Y,X,W,V,U,T,S,R,Q,P,O,N,M,L,K,J,I,H,G,F,E,1,D,C,B,A,9,8,7,6,5,4,3,2".split(',');
let   SIZE = 224;               // try 160 on low-RAM devices
const MIRROR = true;            // selfie view
let   INTERVAL_MS = 250;        // ~4 Hz; increase to 300 if device struggles
const SMOOTH = 15;              // temporal smoothing window
const WARN_THRESH = 0.70;       // low-confidence color cue

// ------------------ UI Elements --------------------
const statusEl  = document.getElementById('status');
const video     = document.getElementById('video');
const runBtn    = document.getElementById('runBtn');
const stopBtn   = document.getElementById('stopBtn');
const predEl    = document.getElementById('pred');
const confEl    = document.getElementById('conf');
const fpsEl     = document.getElementById('fps');
const summaryEl = document.getElementById('summary'); // optional summary card

// ------------------ State --------------------------
let running = false;
let session = null;
let inputName = null;

// Offscreen canvas for preprocessing (single instance)
const canvas = document.createElement('canvas');
let   ctx = null;

// Pre-allocated input buffer/tensor (resized when SIZE changes)
let H = SIZE, W = SIZE, C = 3;
let inBuf = new Float32Array(1 * C * H * W);
let inTensor = new ort.Tensor("float32", inBuf, [1, C, H, W]);

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
  // ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  session = await ort.InferenceSession.create("weights/best.onnx");
  inputName = session.inputNames[0];
}
function prepareCanvas() {
  canvas.width = SIZE; canvas.height = SIZE;
  ctx = canvas.getContext('2d', { willReadFrequently: true });
  // renew buffers for new SIZE if needed
  H = SIZE; W = SIZE;
  inBuf = new Float32Array(1 * C * H * W);
  inTensor = new ort.Tensor("float32", inBuf, [1, C, H, W]);
}

// ------------------ Preprocess ---------------------
function preprocessToNCHW() {
  if (MIRROR) {
    ctx.save(); ctx.translate(SIZE, 0); ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, SIZE, SIZE); ctx.restore();
  } else {
    ctx.drawImage(video, 0, 0, SIZE, SIZE);
  }
  const data = ctx.getImageData(0, 0, SIZE, SIZE).data; // RGBA
  let p = 0, rIdx = 0, gIdx = H*W, bIdx = 2*H*W;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      inBuf[rIdx + p] = data[i]   / 255;
      inBuf[gIdx + p] = data[i+1] / 255;
      inBuf[bIdx + p] = data[i+2] / 255;
      p++;
    }
  }
  return inTensor; // reuse tensor object
}

// ------------------ Math helpers -------------------
function softmax(logits) {
  let m = -Infinity; for (let v of logits) if (v > m) m = v;
  const exps = logits.map(v => Math.exp(v - m));
  const s = exps.reduce((a,b)=>a+b, 0);
  return exps.map(v => v / s);
}
function pushProb(vec) {
  // store a compact copy to avoid holding onto large buffers
  const copy = new Float32Array(vec.length);
  copy.set(vec);
  smoothQ.push(copy);
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
  const tensor = preprocessToNCHW();
  const out = await session.run({ [inputName]: tensor });
  const outName = session.outputNames[0];

  // Avoid Array.from for big outputs; read as typed array directly
  const raw = out[outName].data;
  const logits = (raw instanceof Float32Array || raw instanceof Float64Array) ? raw : Float32Array.from(raw);
  const probs = (Array.prototype.every.call(logits, v => v>=0 && v<=1) &&
                 Math.abs(logits.reduce((a,b)=>a+b,0)-1) < 1e-3)
               ? logits
               : softmax(Array.from(logits));

  pushProb(probs);
  const avg = avgProb() || probs;

  let top = 0; for (let i=1;i<avg.length;i++) if (avg[i] > avg[top]) top = i;
  const label = LABELS[top] || '?';
  const conf  = avg[top];

  // UI
  predEl.textContent = label;
  confEl.textContent = conf.toFixed(2);
  colorByConf(conf);

  // FPS
  const tNow = performance.now();
  const dt = Math.max(1e-3, (tNow - tPrev)/1000); tPrev = tNow;
  fpsEMA = 0.9*fpsEMA + 0.1*(1.0/dt);
  fpsEl.textContent = fpsEMA.toFixed(1);

  // Metrics
  frameCount++; trackLabel(label);
}

// ------------------ Session summary ----------------
function renderSummary() {
  let total = 0, topLabel = '—', topCount = 0;
  for (const [,cnt] of labelHist) total += cnt;
  for (const [k,c] of labelHist) if (c > topCount) { topCount = c; topLabel = k; }
  const share = total ? ((topCount/total)*100).toFixed(1) + '%' : '—';
  const html = `
    <p><span class="badge ok">Session complete</span></p>
    <p><strong>Frames:</strong> ${frameCount}</p>
    <p><strong>Avg FPS (EMA):</strong> ${fpsEl.textContent || '—'}</p>
    <p><strong>Top class:</strong> ${topLabel} <span class="muted">(${share} of frames)</span></p>
    <p><strong>Last confidence:</strong> ${confEl.textContent || '—'}</p>
  `;
  if (summaryEl) summaryEl.innerHTML = html;
  console.log(`[SUMMARY] frames=${frameCount} avg_fps=${fpsEl.textContent} top=${topLabel} share=${share} last_conf=${confEl.textContent}`);
}

// ------------------ Controlled loop (no piling) ----
async function loop() {
  while (running) {
    const t0 = performance.now();
    try { await step(); }
    catch (e) { console.warn('step error:', e); }
    const elapsed = performance.now() - t0;
    const wait = Math.max(0, INTERVAL_MS - elapsed);
    await new Promise(r => setTimeout(r, wait));
  }
}

// ------------------ Controls -----------------------
async function start() {
  if (running) return;
  setStatus('Loading model…', true);
  try {
    if (!video.srcObject) await initCamera();
    if (!session) await initSession();
    if (!ctx || canvas.width !== SIZE) prepareCanvas();
  } catch (e) {
    setStatus('Camera/Model error', false);
    console.error(e);
    return;
  }

  // Auto‑downshift for low‑RAM devices on first run if crash was observed
  try {
    // heuristic: if device has < 4GB RAM, prefer smaller SIZE/interval
    // navigator.deviceMemory is not supported everywhere
    if (navigator.deviceMemory && navigator.deviceMemory < 4) {
      SIZE = 160; INTERVAL_MS = 300; prepareCanvas();
    }
  } catch (_) {}

  resetSessionMetrics();
  running = true;
  setStatus('Running', true);
  loop();
}

function stop() {
  if (!running) return;
  running = false;
  setStatus('Idle', false);
  renderSummary();
}

// Wire up
runBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
setStatus('Idle', false);
