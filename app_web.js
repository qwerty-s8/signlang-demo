
/*const API_URL = "https://signlang-demo.onrender.com/predict";
=======

// app_web.js — Hybrid (Local ONNX + API fallback), hardened for API mode

// ------------------ Endpoints & Modes ------------------
const API_URL = "https://signlang-demo.onrender.com/predict"; // change if needed

function shouldUseAPI() {
  const urlMode = new URLSearchParams(location.search).get("mode");
  if (urlMode === "api") return true;
  if (urlMode === "local") return false;
=======
const API_URL = "https://signlang-demo.onrender.com/predict";
>>>>>>> 4a55cfb4250e4be864ca25b526ce2d3614b7de59

function shouldUseAPI() {
  // Use API if:
  // - device has very low memory, or
  // - WebGL not available for ORT, or
  // - user forced fallback via URL ?mode=api
  const urlMode = new URLSearchParams(location.search).get("mode");
  if (urlMode === "api") return true;
<<<<<<< HEAD
=======
>>>>>>> e572d89 (api)
>>>>>>> 4a55cfb4250e4be864ca25b526ce2d3614b7de59
  try {
    if (navigator.deviceMemory && navigator.deviceMemory < 4) return true;
  } catch(_) {}
  const gl = document.createElement('canvas').getContext('webgl');
<<<<<<< HEAD
=======
<<<<<<< HEAD
  return !gl; // if no WebGL, prefer API
}
let USE_API = shouldUseAPI();
=======
>>>>>>> 4a55cfb4250e4be864ca25b526ce2d3614b7de59
  const webglOK = !!gl;
  return !webglOK; // no GPU → use API
}
let USE_API = shouldUseAPI();
// app_web.js — Browser-only ONNX inference (optimized + polished)
>>>>>>> e572d89 (api)

// ------------------ Configuration ------------------
const LABELS = "Z,Y,X,W,V,U,T,S,R,Q,P,O,N,M,L,K,J,I,H,G,F,E,1,D,C,B,A,9,8,7,6,5,4,3,2".split(',');
let   SIZE = 160;
const MIRROR = true;
let   INTERVAL_MS = 300;
const SMOOTH = 10;
const WARN_THRESH = 0.70;

// API sending config
const SEND_SIZE = 160;
const SEND_MS   = 300;

// ------------------ UI Elements --------------------
const statusEl  = document.getElementById('status');
const video     = document.getElementById('video');
const runBtn    = document.getElementById('runBtn');
const stopBtn   = document.getElementById('stopBtn');
const predEl    = document.getElementById('pred');
const confEl    = document.getElementById('conf');
const fpsEl     = document.getElementById('fps');
const summaryEl = document.getElementById('summary');

// Optional: show current mode in status
function setModeTag() {
  const tag = USE_API ? 'API' : 'Local';
  statusEl.setAttribute('title', `Mode: ${tag}`);
}

// ------------------ State --------------------------
let running = false;
let session = null;
let inputName = null;

const canvas = document.createElement('canvas');
let   ctx = null;

let H = SIZE, W = SIZE, C = 3;
let inBuf = new Float32Array(1 * C * H * W);
let inTensor = null;

let fpsEMA = 0.0, tPrev = performance.now();
const smoothQ = [];
let frameCount = 0;
const labelHist = new Map();

// API capture canvas
const cSend = document.createElement('canvas');
cSend.width = SEND_SIZE; cSend.height = SEND_SIZE;
const cxSend = cSend.getContext('2d', { willReadFrequently: true });

// ------------------ Helpers ------------------------
function setStatus(txt, on=false) {
  statusEl.textContent = txt;
  statusEl.classList.toggle('on', on);
  statusEl.classList.toggle('off', !on);
}
function colorByConf(v) {
  const low = Number(v) < WARN_THRESH;
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
  if (summaryEl) {
    summaryEl.innerHTML = '<p class="muted">Running… a summary will appear here when you stop.</p>';
  }
}

// ------------------ Camera -------------------------
async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await video.play();
}

// ------------------ API path -----------------------
async function sendFrameJSON() {
  cxSend.drawImage(video, 0, 0, SEND_SIZE, SEND_SIZE);
  const dataUrl = cSend.toDataURL('image/jpeg', 0.7);
  const res = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataUrl })
  });
  if (!res.ok) {
    // Bubble up useful info in console without breaking the loop
    const txt = await res.text().catch(()=> String(res.status));
    throw new Error(`API ${res.status}: ${txt}`);
  }
  return res.json();
}

async function loopAPI() {
  while (running) {
    const t0 = performance.now();
    try {
      const r = await sendFrameJSON();
      if (r && r.ok) {
        const label = r.label ?? '—';
        const conf  = Number(r.conf ?? 0);
        predEl.textContent = label;
        confEl.textContent = conf.toFixed(2);
        colorByConf(conf);
        frameCount++;
        labelHist.set(label, (labelHist.get(label)||0)+1);
      } else {
        console.warn('Unexpected API response:', r);
      }
    } catch (e) {
      console.warn('API error:', e.message || e);
    }

    const tNow = performance.now();
    const dt = Math.max(1e-3, (tNow - tPrev)/1000);
    tPrev = tNow;
    fpsEMA = 0.9*fpsEMA + 0.1*(1.0/dt);
    fpsEl.textContent = fpsEMA.toFixed(1);

    const elapsed = performance.now() - t0;
    await new Promise(r => setTimeout(r, Math.max(0, SEND_MS - elapsed)));
  }
}

// ------------------ Local ONNX path ----------------
async function initSession() {
  if (typeof ort === 'undefined') throw new Error('onnxruntime-web not loaded');
  const EP = [{ name: 'webgl' }, { name: 'wasm' }];
  session = await ort.InferenceSession.create("weights/best.onnx", { executionProviders: EP });
  inputName = session.inputNames[0];
  inTensor = new ort.Tensor("float32", inBuf, [1, C, H, W]);
}
function prepareCanvas() {
  canvas.width = SIZE; canvas.height = SIZE;
  ctx = canvas.getContext('2d', { willReadFrequently: true });
  H = SIZE; W = SIZE;
  inBuf = new Float32Array(1 * C * H * W);
  if (typeof ort !== 'undefined') {
    inTensor = new ort.Tensor("float32", inBuf, [1, C, H, W]);
  }
}
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
  return inTensor;
}
function softmax(logits) {
  let m = -Infinity; for (let v of logits) if (v > m) m = v;
  const exps = logits.map(v => Math.exp(v - m));
  const s = exps.reduce((a,b)=>a+b, 0);
  return exps.map(v => v / s);
}
function pushProb(vec) {
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

async function step() {
  const tensor = preprocessToNCHW();
  const out = await session.run({ [inputName]: tensor });
  const outName = session.outputNames[0];
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

  predEl.textContent = label;
  confEl.textContent = conf.toFixed(2);
  colorByConf(conf);

  const tNow = performance.now();
  const dt = Math.max(1e-3, (tNow - tPrev)/1000); tPrev = tNow;
  fpsEMA = 0.9*fpsEMA + 0.1*(1.0/dt);
  fpsEl.textContent = fpsEMA.toFixed(1);

  frameCount++; trackLabel(label);
}

async function loopLocal() {
  while (running) {
    const t0 = performance.now();
    try { await step(); } catch (e) { console.warn('step error:', e); }
    const elapsed = performance.now() - t0;
    await new Promise(r => setTimeout(r, Math.max(0, INTERVAL_MS - elapsed)));
  }
}

// ------------------ Summary ------------------------
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

// ------------------ Controls -----------------------
async function start() {
  if (running) return;
  setModeTag();
  setStatus('Starting…', true);
  try {
    if (!video.srcObject) await initCamera();
  } catch (e) {
    setStatus('Camera error', false);
    console.error(e);
    return;
  }

  // Light auto-downshift
  try {
    if (navigator.deviceMemory && navigator.deviceMemory < 4) {
      SIZE = 160; INTERVAL_MS = 300; prepareCanvas();
    }
  } catch(_) {}

  resetSessionMetrics();
  running = true;

  if (USE_API) {
    setStatus('Running (API)', true);
    loopAPI();
    return;
  }

  // Try local ONNX; if it fails, fallback to API
  setStatus('Loading model…', true);
  try {
    if (!session) await initSession();
    if (!ctx || canvas.width !== SIZE) prepareCanvas();
    setStatus('Running (Local)', true);
    loopLocal();
  } catch (e) {
    console.warn('Local init failed, falling back to API:', e);
    USE_API = true;
    setStatus('Running (API fallback)', true);
    loopAPI();
  }
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
setStatus('Idle', false);*/
// app_web.js — Pure API client (forced), with FPS and summary

// ------------- Config -------------
const API_URL   = "https://signlang-demo.onrender.com/predict"; // <-- change if needed
const SEND_SIZE = 160;
const SEND_MS   = 300;
const WARN_THRESH = 0.70;

// ------------- UI refs ------------
const statusEl = document.getElementById('status');
const video    = document.getElementById('video');
const runBtn   = document.getElementById('runBtn');
const stopBtn  = document.getElementById('stopBtn');
const predEl   = document.getElementById('pred');
const confEl   = document.getElementById('conf');
const fpsEl    = document.getElementById('fps');
const summaryEl= document.getElementById('summary');

// ------------- State --------------
let running = false;
let fpsEMA = 0.0, tPrev = performance.now();
let frameCount = 0;
const labelHist = new Map();

// Offscreen canvas for capture/resize
const cSend = document.createElement('canvas');
cSend.width = SEND_SIZE; cSend.height = SEND_SIZE;
const cxSend = cSend.getContext('2d', { willReadFrequently: true });

// ------------- Helpers ------------
function setStatus(txt, on=false) {
  statusEl.textContent = txt;
  statusEl.classList.toggle('on', on);
  statusEl.classList.toggle('off', !on);
  statusEl.title = "Mode: API";
}
function colorByConf(v) {
  const low = Number(v) < WARN_THRESH;
  const c = low ? '#f4bf4f' : '#e9ecf1';
  predEl.style.color = c;
  confEl.style.color = c;
}
function resetMetrics() {
  fpsEMA = 0.0; tPrev = performance.now();
  frameCount = 0; labelHist.clear();
  predEl.textContent = '—'; confEl.textContent = '—'; fpsEl.textContent = '—';
  if (summaryEl) summaryEl.innerHTML = '<p class="muted">Running… a summary will appear here when you stop.</p>';
}

// ------------- Camera -------------
async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await video.play();
}

// ------------- Networking ---------
async function sendFrameJSON() {
  if (!video.videoWidth) throw new Error('video not ready');
  cxSend.drawImage(video, 0, 0, SEND_SIZE, SEND_SIZE);
  const blob = await new Promise(r => cSend.toBlob(r, 'image/jpeg', 0.7));
  const fd = new FormData();
  fd.append('file', blob, 'frame.jpg'); // field name must match FastAPI parameter
  const res = await fetch(API_URL, { method: 'POST', body: fd });
  if (!res.ok) {
    const txt = await res.text().catch(()=> String(res.status));
    throw new Error(`API ${res.status}: ${txt}`);
  }
  return res.json();
}


// ------------- Main loop ----------
async function loopAPI() {
  while (running) {
    const t0 = performance.now();
    try {
      const r = await sendFrameJSON();
      console.debug('predict response:', r); // temporary: verify keys once
      const ok    = (r.ok === true) || (r.success === true) || ('label' in r) || ('pred' in r);
      if (ok) {
        const label = (r.label ?? r.pred ?? r.class ?? '—');
        const conf  = Number(r.conf ?? r.score ?? r.prob ?? r.probability ?? 0);
        predEl.textContent = String(label);
        confEl.textContent = (isFinite(conf) ? conf.toFixed(2) : '—');
        colorByConf(conf);
        frameCount++; labelHist.set(String(label), (labelHist.get(String(label))||0)+1);
      }else {
        console.warn('Unexpected API response:', r);
      }
    } catch (e) {
      console.warn('API error:', e.message || e);
    }

    const tNow = performance.now();
    const dt = Math.max(1e-3, (tNow - tPrev)/1000); tPrev = tNow;
    fpsEMA = 0.9*fpsEMA + 0.1*(1.0/dt);
    fpsEl.textContent = fpsEMA.toFixed(1);

    const elapsed = performance.now() - t0;
    await new Promise(r => setTimeout(r, Math.max(0, SEND_MS - elapsed)));
  }
}

// ------------- Summary ------------
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
}

// ------------- Controls ----------
async function start() {
  if (running) return;
  setStatus('Starting…', true);
  try { if (!video.srcObject) await initCamera(); }
  catch (e) { setStatus('Camera error', false); console.error(e); return; }

  resetMetrics();
  running = true;
  setStatus('Running (API)', true);
  loopAPI();
}
function stop() {
  if (!running) return;
  running = false;
  setStatus('Idle', false);
  renderSummary();
}
runBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
setStatus('Idle', false);
