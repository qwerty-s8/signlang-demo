// app_web.js — API-powered client (stable, low-RAM)

// ------------------ Configuration ------------------
const API_URL    = "https://signlang-demo.onrender.com/predict"; // change if needed
const SEND_SIZE  = 160;    // 160x160 JPEG to server
const SEND_MS    = 300;    // ~3.3 Hz
const WARN_THRESH = 0.70;  // low-confidence color cue

// ------------------ UI Elements --------------------
const statusEl  = document.getElementById('status');
const video     = document.getElementById('video');
const runBtn    = document.getElementById('runBtn');
const stopBtn   = document.getElementById('stopBtn');
const predEl    = document.getElementById('pred');
const confEl    = document.getElementById('conf');
const fpsEl     = document.getElementById('fps');
const summaryEl = document.getElementById('summary'); // optional

// ------------------ State --------------------------
let running = false;
let fpsEMA = 0.0, tPrev = performance.now();
let frameCount = 0;
const labelHist = new Map();

// Offscreen canvas used for capture/resize
const cSend = document.createElement('canvas');
cSend.width = SEND_SIZE; cSend.height = SEND_SIZE;
const cxSend = cSend.getContext('2d', { willReadFrequently: true });

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

// ------------------ Networking ---------------------
async function sendFrameJSON() {
  // Draw frame and encode to base64 JPEG
  cxSend.drawImage(video, 0, 0, SEND_SIZE, SEND_SIZE);
  const dataUrl = cSend.toDataURL('image/jpeg', 0.7);
  const res = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataUrl })
  });
  return res.json();
}

// ------------------ Loop ---------------------------
async function loop() {
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
        labelHist.set(label, (labelHist.get(label) || 0) + 1);
      }
    } catch (e) {
      // Non-fatal; keep UI responsive
      console.warn('predict error:', e);
    }

    // FPS
    const tNow = performance.now();
    const dt = Math.max(1e-3, (tNow - tPrev)/1000);
    tPrev = tNow;
    fpsEMA = 0.9*fpsEMA + 0.1*(1.0/dt);
    fpsEl.textContent = fpsEMA.toFixed(1);

    const elapsed = performance.now() - t0;
    const wait = Math.max(0, SEND_MS - elapsed);
    await new Promise(r => setTimeout(r, wait));
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
  setStatus('Connecting…', true);
  try {
    if (!video.srcObject) await initCamera();
  } catch (e) {
    setStatus('Camera error', false);
    console.error(e);
    return;
  }
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
