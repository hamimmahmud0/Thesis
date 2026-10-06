"""Zoomable / pannable image canvas shared by the box editor (editor.py) and the image viewer (viewer.py).

Mouse: wheel = zoom around the cursor, middle or right button drag = pan. The left button belongs to the owner.
Overlays (boxes, labels) are drawn by an `overlay(canvas)` callback in *image* coordinates.
"""
import math
import tkinter as tk
import zlib

import cv2
from PIL import Image, ImageTk

PALETTE = ["#ff5c5c", "#4cc9f0", "#7bd88f", "#ffb347", "#c08cff", "#ff8fd8", "#f2e94e", "#4d96ff",
           "#2ec4b6", "#ff7f50", "#9ccc65", "#b0bec5"]


SCALE = 1.0   # UI scale factor, set from --ui-scale before any widget is built


def S(n):
    """Scale a size in px / pt by the UI scale."""
    return max(1, round(n * SCALE))


def class_color(name, names=()):
    names = list(names)
    i = names.index(name) if name in names else zlib.crc32(str(name).encode())
    return PALETTE[i % len(PALETTE)]


class View:
    """Image <-> canvas mapping: canvas = (image - origin) * scale. Pure maths, no Tk."""

    def __init__(self, scale=1.0, x0=0.0, y0=0.0):
        self.scale, self.x0, self.y0 = scale, x0, y0

    def to_canvas(self, x, y):
        return (x - self.x0) * self.scale, (y - self.y0) * self.scale

    def to_img(self, cx, cy):
        return cx / self.scale + self.x0, cy / self.scale + self.y0

    def zoom_at(self, cx, cy, factor, lo, hi):
        ix, iy = self.to_img(cx, cy)
        self.scale = min(hi, max(lo, self.scale * factor))
        self.x0, self.y0 = ix - cx / self.scale, iy - cy / self.scale

    def pan_px(self, dx, dy):
        self.x0 -= dx / self.scale
        self.y0 -= dy / self.scale

    def fit_rect(self, rect, W, H, margin=0.0):
        """Scale and centre so that rect = (x0, y0, x1, y1) fills the canvas, with a relative margin."""
        x0, y0, x1, y1 = rect
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        self.scale = min(W / (w * (1 + 2 * margin)), H / (h * (1 + 2 * margin)))
        self.x0 = (x0 + x1) / 2 - W / 2 / self.scale
        self.y0 = (y0 + y1) / 2 - H / 2 / self.scale

    def clamp(self, iw, ih, W, H, keep=100):
        """Keep at least `keep` canvas px of the image visible."""
        s = self.scale
        self.x0 = min(max(self.x0, (keep - W) / s), iw - keep / s)
        self.y0 = min(max(self.y0, (keep - H) / s), ih - keep / s)


class BoxCanvas(tk.Canvas):
    def __init__(self, master, overlay=None, **kw):
        super().__init__(master, bg="#101010", highlightthickness=0, **kw)
        self.view, self.img, self.overlay = View(), None, overlay
        self._photo, self._want_fit, self._pan = None, None, None
        self.bind("<Configure>", self._on_configure)
        self.bind("<MouseWheel>", lambda e: self._wheel(e.x, e.y, 1.25 if e.delta > 0 else 0.8))
        self.bind("<Button-4>", lambda e: self._wheel(e.x, e.y, 1.25))
        self.bind("<Button-5>", lambda e: self._wheel(e.x, e.y, 0.8))
        for b in (2, 3):
            self.bind(f"<ButtonPress-{b}>", self._pan_start)
            self.bind(f"<B{b}-Motion>", self._pan_move)

    # ---- geometry
    @property
    def size(self):
        return self.winfo_width(), self.winfo_height()

    def to_img(self, cx, cy):
        return self.view.to_img(cx, cy)

    def to_canvas(self, x, y):
        return self.view.to_canvas(x, y)

    def fit_scale(self):
        W, H = self.size
        ih, iw = self.img.shape[:2]
        return min(W / iw, H / ih)

    def set_image(self, img, rect=None, margin=0.0):
        """Show img (BGR ndarray). rect = (x0, y0, x1, y1) in image px to fit, default whole image."""
        self.img = img
        ih, iw = img.shape[:2]
        self._want_fit = (rect or (0, 0, iw, ih), margin)
        if self.size[0] > 50:
            self.refresh()

    def _apply_fit(self):
        rect, margin = self._want_fit
        self._want_fit = None
        W, H = self.size
        self.view.fit_rect(rect, W, H, margin)
        self._limit()

    def _limit(self):
        W, H = self.size
        ih, iw = self.img.shape[:2]
        self.view.scale = min(max(self.view.scale, self.fit_scale() * 0.5), 24.0)
        self.view.clamp(iw, ih, W, H)

    # ---- events
    def _on_configure(self, _e):
        if self.img is not None:
            self.refresh()

    def _wheel(self, x, y, factor):
        if self.img is None:
            return
        self.view.zoom_at(x, y, factor, self.fit_scale() * 0.5, 24.0)
        self._limit()
        self.refresh()

    def _pan_start(self, e):
        self._pan = (e.x, e.y)

    def _pan_move(self, e):
        if self.img is None or self._pan is None:
            return
        self.view.pan_px(e.x - self._pan[0], e.y - self._pan[1])
        self._pan = (e.x, e.y)
        self._limit()
        self.refresh()

    # ---- drawing
    def refresh(self):
        if self.img is None or self.size[0] < 50:
            return
        if self._want_fit:
            self._apply_fit()
        self._draw_image()
        self.draw_overlay()

    def _draw_image(self):
        self.delete("img")
        W, H = self.size
        v, s = self.view, self.view.scale
        ih, iw = self.img.shape[:2]
        ix0, iy0 = max(0, math.floor(v.x0)), max(0, math.floor(v.y0))
        ix1, iy1 = min(iw, math.ceil(v.x0 + W / s)), min(ih, math.ceil(v.y0 + H / s))
        if ix1 <= ix0 or iy1 <= iy0:
            return
        roi = self.img[iy0:iy1, ix0:ix1]
        size = (max(1, round((ix1 - ix0) * s)), max(1, round((iy1 - iy0) * s)))
        interp = cv2.INTER_AREA if s < 1 else (cv2.INTER_NEAREST if s >= 6 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(cv2.resize(roi, size, interpolation=interp), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        cx, cy = v.to_canvas(ix0, iy0)
        self.create_image(cx, cy, image=self._photo, anchor="nw", tags="img")
        self.tag_lower("img")

    def draw_overlay(self):
        self.delete("ov")
        if self.overlay and self.img is not None:
            self.overlay(self)

    # ---- helpers for overlay callbacks (image coordinates in, canvas items out)
    def box(self, b, color, width=2, dash=None, fill=None):
        """b = (x0, y0, x1, y1) in image px."""
        (x0, y0), (x1, y1) = self.to_canvas(b[0], b[1]), self.to_canvas(b[2], b[3])
        kw = {"stipple": "gray25"} if fill else {}
        return self.create_rectangle(x0, y0, x1, y1, outline=color, width=width, dash=dash, fill=fill or "",
                                     tags="ov", **kw)

    def text(self, x, y, s, color="#fff", anchor="sw", size=13):
        """Label with a dark backing, anchored at image point (x, y)."""
        cx, cy = self.to_canvas(x, y)
        t = self.create_text(cx, cy, text=s, fill=color, anchor=anchor, font=("TkDefaultFont", S(size), "bold"),
                             tags="ov")
        bb = self.bbox(t)
        if bb:
            r = self.create_rectangle(bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1, fill="#000", outline="", tags="ov")
            self.tag_lower(r, t)
        return t
