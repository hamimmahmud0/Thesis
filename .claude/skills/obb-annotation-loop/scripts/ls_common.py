"""Shared Label Studio helpers: JWT refresh-token auth, retrying requests, rotated-rectangle <-> polygon geometry.

Auth: pass the path of a file holding a Label Studio *refresh* token (Account & Settings -> Access Token). The file is only read, never
written; create it with umask 077 on the VM and delete it (shred -u) when the job ends. Access tokens expire in minutes, so every call
re-mints one when needed.

Rotated rectangle convention (verified against Label Studio's own YOLO_OBB export): x, y = top-left corner as percent of image w/h,
width/height = percent of image w/h, rotation = degrees clockwise about (x, y), applied in pixel space.
"""
import math
import threading
import time

import requests


class LS:
    def __init__(self, url, token_file):
        self.url = url.rstrip('/')
        self.refresh = open(token_file).read().strip()
        self._lock = threading.Lock()
        self._tok, self._t = None, 0.0

    def access(self, force=False):
        with self._lock:
            if force or self._tok is None or time.time() - self._t > 150:
                r = requests.post(f'{self.url}/api/token/refresh', json={'refresh': self.refresh}, timeout=30)
                r.raise_for_status()
                self._tok, self._t = r.json()['access'], time.time()
            return self._tok

    def call(self, method, path, **kw):
        """Retrying request (path is relative, e.g. /api/projects). 401 -> re-mint token; 5xx / network errors -> back off."""
        last = None
        for attempt in range(5):
            try:
                r = requests.request(method, self.url + path, headers={'Authorization': f'Bearer {self.access(attempt == 1)}'}, timeout=kw.pop('timeout', 300), **kw)
            except requests.RequestException as e:
                last = e; time.sleep(2 * (attempt + 1)); continue
            if r.status_code == 401:
                self.access(True); continue
            if r.status_code >= 500:
                last = RuntimeError(f'{r.status_code} {r.text[:120]}'); time.sleep(2 * (attempt + 1)); continue
            return r
        raise RuntimeError(f'{method} {path} failed: {last}')

    def projects(self):
        out, page = [], 1
        while True:
            r = self.call('GET', f'/api/projects?page_size=100&page={page}')
            r.raise_for_status()
            d = r.json()
            res = d['results'] if isinstance(d, dict) else d
            out += res
            if not isinstance(d, dict) or not d.get('next'):
                return out
            page += 1


def rect_from_poly(poly, W, H):
    """4-corner polygon (pixels) -> Label Studio rotated-rectangle value + round-trip error (px) + (w, h) in pixels."""
    p = [tuple(map(float, q)) for q in poly]
    e1 = (p[1][0] - p[0][0], p[1][1] - p[0][1]); e2 = (p[3][0] - p[0][0], p[3][1] - p[0][1])
    if e1[0] * e2[1] - e1[1] * e2[0] < 0:  # counter-clockwise order -> flip so rotation stays clockwise-positive
        p = [p[0], p[3], p[2], p[1]]; e1 = (p[1][0] - p[0][0], p[1][1] - p[0][1]); e2 = (p[3][0] - p[0][0], p[3][1] - p[0][1])
    w, h = math.hypot(*e1), math.hypot(*e2); th = math.atan2(e1[1], e1[0]); c, s = math.cos(th), math.sin(th)
    rec = [(p[0][0] + c * dx - s * dy, p[0][1] + s * dx + c * dy) for dx, dy in ((0, 0), (w, 0), (w, h), (0, h))]
    err = max(min(math.hypot(a - b[0], bb - b[1]) for b in p) for a, bb in rec)
    return {'x': p[0][0] / W * 100, 'y': p[0][1] / H * 100, 'width': w / W * 100, 'height': h / H * 100, 'rotation': math.degrees(th)}, err, (w, h)


def poly_from_rect(v, W, H):
    """Label Studio rotated-rectangle value -> 4 corner points in pixels (clockwise from the top-left corner)."""
    x, y = v['x'] / 100 * W, v['y'] / 100 * H
    w, h = v['width'] / 100 * W, v['height'] / 100 * H
    th = math.radians(v.get('rotation', 0.0)); c, s = math.cos(th), math.sin(th)
    return [(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((0, 0), (w, 0), (w, h), (0, h))]
