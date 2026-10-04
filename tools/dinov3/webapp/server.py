#!/usr/bin/env python3
"""Stdlib server for the drop selector: serves public/ and POST /api/submit -> Telegram.
Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, PORT (default 5675), HOST (default 127.0.0.1), DEFAULT_PAGE (e.g. /b1.html), ACCESS_CODE (optional)."""
import json, os, re, time, urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
ID_RE = re.compile(r"^[RM]\d{3}$"); last = [0.0]

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=ROOT, **k)
    def _json(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.rstrip("/") in ("/inspect", "/inspect/index.html"): self.path = "/inspect.html"
        if self.path.rstrip("/") == "/b1": self.path = "/b1.html"
        if self.path == "/" and os.environ.get("DEFAULT_PAGE"): self.path = os.environ["DEFAULT_PAGE"]
        super().do_GET()
    def do_POST(self):
        if self.path != "/api/submit": return self._json(404, {"error": "not found"})
        try: body = json.loads(self.rfile.read(min(int(self.headers.get("content-length", 0)), 20000)))
        except Exception: return self._json(400, {"error": "bad json"})
        code = os.environ.get("ACCESS_CODE")
        if code and body.get("code") != code: return self._json(403, {"error": "wrong access code"})
        ids = body.get("dropped")
        if not isinstance(ids, list) or len(ids) > 200 or not all(isinstance(i, str) and ID_RE.match(i) for i in ids):
            return self._json(400, {"error": "invalid ids"})
        if time.time() - last[0] < 5: return self._json(429, {"error": "too fast, wait a few seconds"})
        last[0] = time.time(); ids = sorted(set(ids))
        r, m = [i for i in ids if i[0] == "R"], [i for i in ids if i[0] == "M"]
        text = (f"Drop list submitted\nRickshaw: drop {len(r)}, keep {100-len(r)}\n{', '.join(r) or '-'}\n\n"
                f"Motorcycle: drop {len(m)}, keep {100-len(m)}\n{', '.join(m) or '-'}")
        try:
            req = urllib.request.Request(f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage",
                data=json.dumps({"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text}).encode(),
                headers={"content-type": "application/json"})
            urllib.request.urlopen(req, timeout=15).read()
        except Exception as e:
            return self._json(502, {"error": "telegram failed"})
        with open(os.path.join(os.path.dirname(ROOT), "submissions.log"), "a") as f:
            f.write(json.dumps({"t": time.time(), "dropped": ids}) + "\n")
        self._json(200, {"ok": True, "dropped": len(ids)})

if __name__ == "__main__":
    ThreadingHTTPServer((os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", 5675))), H).serve_forever()
