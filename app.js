/*const modal = { el:null, show(){this.el.style.display='flex'}, hide(){this.el.style.display='none'} };

function openLocal(path){
// Attempt to navigate to local helper .bat; some browsers prompt for confirmation.
window.location.href = path;
}

document.addEventListener('DOMContentLoaded', ()=>{
modal.el = document.getElementById('run-modal');
document.getElementById('btn-run').addEventListener('click', (e)=>{
e.preventDefault();
modal.show();
});
document.getElementById('btn-close').addEventListener('click', ()=>modal.hide());

document.getElementById('btn-open-folder').addEventListener('click', ()=>{
openLocal('launch/open_demo_folder.bat');
});
document.getElementById('btn-run-helper').addEventListener('click', ()=>{
openLocal('launch/run_demo.bat');
});
document.getElementById('btn-copy-path').addEventListener('click', async ()=>{
try{
await navigator.clipboard.writeText('SignLangDemo\start_demo.bat');
alert('Path copied. Open this folder and double‑click start_demo.bat');
}catch(_){ alert('Copy failed. Path: SignLangDemo\start_demo.bat'); }
});
});*/

const statusEl = document.getElementById('status');
const demoCard = document.getElementById('demo');
const demoPill = document.getElementById('demoPill');
const runBtn = document.getElementById('runBtn');
const stopBtn = document.getElementById('stopBtn');
const predEl = document.getElementById('pred');
const confEl = document.getElementById('conf');
const fpsEl  = document.getElementById('fps');

let running = false;
let statsTimer = null;

function startStats() {
  if (statsTimer) return;
  statsTimer = setInterval(async () => {
    try {
      const r = await fetch('http://127.0.0.1:8787/stats');
      if (!r.ok) return;
      const s = await r.json();
      predEl.textContent = s.label || '—';
      confEl.textContent = (s.conf ?? 0).toFixed(2);
      fpsEl.textContent  = (s.fps  ?? 0).toFixed(1);
      console.log(`[STAT] ${s.label} conf=${s.conf} fps=${s.fps}`);
    } catch (_e) {
      // ignore transient errors while starting/stopping
    }
  }, 250); // 4 Hz
}
function stopStats() {
  if (statsTimer) { clearInterval(statsTimer); statsTimer = null; }
  predEl.textContent = '—';
  confEl.textContent = '—';
  fpsEl.textContent  = '—';
}




function setStatus(txt) {
  statusEl.textContent = `Status: ${txt}`;
}
function setActive(on) {
  if (on) {
    demoCard.classList.add('active');
    demoPill.textContent = 'Running';
    demoPill.classList.add('on');
    demoPill.classList.remove('off');
  } else {
    demoCard.classList.remove('active');
    demoPill.textContent = 'Inactive';
    demoPill.classList.remove('on');
    demoPill.classList.add('off');
  }
}

/*runBtn.addEventListener('click', () => {
  if (running) return;
  // Attempt to open the batch file located alongside index.html
  // Browser may prompt; confirm to allow.
  //window.location.href = 'start_demo.bat';
  window.location.href = 'start_demo.vbs';
  running = true;
  setActive(true);
  setStatus('Launching demo (check camera window)…');
  // Placeholders for UI; actual values appear in the Python window overlay
  predEl.textContent = '—';
  confEl.textContent = '—';
  fpsEl.textContent  = '—';
});*/
runBtn.addEventListener('click', async () => {
  if (running) return;
  try {
    const r = await fetch('http://127.0.0.1:8787/start', { method: 'POST' });
    console.log('start status', r.status);
    if (!r.ok) throw new Error('start failed');
    running = true;
    setActive(true);
    setStatus('Launching demo (check camera window)…');
    predEl.textContent = '—';
    confEl.textContent = '—';
    fpsEl.textContent  = '—';
    startStats();
  } catch (e) {
    console.error(e);
    alert('Could not reach local server at 127.0.0.1:8787. Is server.py running?');
  }
});





stopBtn.addEventListener('click', () => {
  if (!running) {
    alert('Demo is not running.');
    return;
  }
  alert("Press 'q' in the camera window to quit the demo.");
  setActive(false);
  setStatus('Idle');
  running = false;
  stopStats();
});

document.addEventListener('DOMContentLoaded', () => {
  fetch('http://127.0.0.1:8787/health')
    .then(r => r.json())
    .then(t => {
      console.log('health:', t);
      setStatus('Ready');
    })
    .catch(e => {
      console.warn('health check failed', e);
      setStatus('Server not reachable');
    });
});
