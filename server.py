# server.py
from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess, os, threading, json, sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def launch():
    py = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')
    script = os.path.join(ROOT, 'webcam_onnx_cls.py')
    args = [
        'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
        'Start-Process', '-FilePath', py,
        '-ArgumentList',
        f'"{script}", "--model", ".\\weights\\best.onnx", "--imgsz", "224", "--camera", "0", '
        f'"--labels", "Z,Y,X,W,V,U,T,S,R,Q,P,O,N,M,L,K,J,I,H,G,F,E,1,D,C,B,A,9,8,7,6,5,4,3,2", '
        f'"--smooth", "15", "--conf_thresh", "0.70", "--mirror"',
        '-WorkingDirectory', ROOT
    ]
    print('Launching python via PowerShell', flush=True)
    subprocess.Popen(args, cwd=ROOT)

class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _send(self, code=200, data=None):
        self.send_response(code)
        self._cors()
        if data is None:
            self.end_headers()
            return
        body = json.dumps(data).encode('utf-8')
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stdout.write(("%s - - %s\n" % (self.client_address[0], fmt % args)))
        sys.stdout.flush()

    def do_OPTIONS(self):
        self._send(204)

    def do_POST(self):
        if self.path == '/start':
            print('POST /start received', flush=True)
            threading.Thread(target=launch, daemon=True).start()
            self._send(200, {"ok": True})
        else:
            self._send(404, {"ok": False, "error": "Not found"})

    def do_GET(self):
        if self.path == '/health':
            self._send(200, {"status": "ok"})
            return
        elif self.path == '/stats':
            try:
                p = os.path.join(ROOT, 'latest_stats.json')
                if os.path.exists(p):
                    with open(p, 'rb') as f:
                        data = f.read()
                    self.send_response(200); self._cors()
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._send(200, {"label": "-", "conf": 0.0, "fps": 0.0})
            except Exception as e:
                self._send(500, {"ok": False, "error": str(e)})
            return
        else:
            self._send(404, {"ok": False, "error": "Not found"})
            return

if __name__ == '__main__':
    print("Server listening on http://127.0.0.1:8787", flush=True)
    HTTPServer(('127.0.0.1', 8787), H).serve_forever()
