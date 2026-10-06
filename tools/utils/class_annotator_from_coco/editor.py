"""Box editor: re-annotate one detection (class and bounding box) as one or several boxes.

Opened from review.py with R. The original detection is shown dashed yellow, neighbouring detections thin grey.

Mouse   drag on empty space      draw a new box
        click a box              select it;  drag inside = move,  drag a handle = resize
        wheel / middle-drag      zoom / pan
Keys    class key                set the class of the selected box (and of boxes drawn next)
        Delete / Backspace       remove the selected box
        Tab / Shift+Tab          select next / previous box
        arrows (+Shift)          nudge the selected box by 1 px (10 px)
        R                        reset to the original detection
        N                        new class
        Enter                    save (no boxes = reject the detection)
        Esc                      cancel, keep the old decision
"""
import tkinter as tk
from tkinter import simpledialog

from boxcanvas import BoxCanvas, S, class_color

HANDLE_PX = 9      # grab distance for resize handles, in canvas px
MIN_DRAG_PX = 6    # a press-release shorter than this is a click, not a new box
MIN_BOX = 2        # smallest box edge kept, in image px
BG, FG = "#1c1c1c", "#dddddd"


def xywh_to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def xyxy_to_xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


def norm(b):
    return [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])]


def handle_points(b):
    """Resize handles of box b = (x0, y0, x1, y1): name -> point."""
    xm, ym = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return {"nw": (b[0], b[1]), "n": (xm, b[1]), "ne": (b[2], b[1]), "e": (b[2], ym),
            "se": (b[2], b[3]), "s": (xm, b[3]), "sw": (b[0], b[3]), "w": (b[0], ym)}


def resize_box(b, handle, p):
    b = list(b)
    if "n" in handle:
        b[1] = p[1]
    if "s" in handle:
        b[3] = p[1]
    if "w" in handle:
        b[0] = p[0]
    if "e" in handle:
        b[2] = p[0]
    return b


def pick_box(boxes, p, prefer=None):
    """Index of the box containing p: the preferred one if it contains p, else the smallest."""
    hit = [i for i, b in enumerate(boxes) if b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]]
    if not hit:
        return None
    if prefer in hit:
        return prefer
    return min(hit, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))


def printable(e):
    return e.char if e.char and e.char.isprintable() and e.char != " " else ""


class BoxEditor(tk.Frame):
    def __init__(self, master, keys, new_class, done):
        """keys: live dict class name -> shortcut. new_class(raw) -> (name | None, message).
        done(result): list of {"bbox": [x, y, w, h], "class": name}, or None when cancelled."""
        super().__init__(master, bg=BG)
        self.keys, self.new_class, self.done = keys, new_class, done
        self.top = tk.Label(self, bg="#2a1f45", fg="#e6d8ff", anchor="w", justify="left", font=("TkDefaultFont", S(14), "bold"), padx=10, pady=5)
        self.top.pack(fill="x")
        self.canvas = BoxCanvas(self, overlay=self._overlay)
        self.canvas.pack(fill="both", expand=True)
        self.legend = tk.Label(self, bg=BG, fg="#ffc864", anchor="w", justify="left", padx=10, pady=3)
        self.legend.pack(fill="x")
        self.hint = tk.Label(self, bg=BG, fg="#5aff78", anchor="w", justify="left", padx=10, pady=4,
                             text="drag = new box   click = select   drag box/handle = move/resize   class key = set class   "
                                  "Del = remove   Tab = next box   R = reset   N = new class   Enter = save   Esc = cancel")
        self.hint.pack(fill="x")
        self.bind("<Configure>", lambda e: [w.configure(wraplength=max(200, e.width - 24)) for w in (self.top, self.legend, self.hint)])
        c = self.canvas
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<B1-Motion>", self._drag)
        c.bind("<ButtonRelease-1>", self._release)
        self.boxes, self.sel, self.drag_state, self.msg = [], None, None, ""

    # ---- session
    def start(self, img, ann, boxes, default_cls, neighbours=(), title=""):
        """boxes: [{"bbox": xywh, "class": name}] to edit; ann: the detection being re-annotated."""
        self.img, self.ann, self.title = img, ann, title
        self.orig = xywh_to_xyxy(ann["bbox"])
        self.neigh = [xywh_to_xyxy(b) for b in neighbours]
        self.boxes = [{"b": xywh_to_xyxy(b["bbox"]), "cls": b["class"]} for b in boxes]
        self.sel = 0 if self.boxes else None
        self.default, self.drag_state, self.msg = default_cls, None, ""
        x0, y0, x1, y1 = self.orig
        m = max(x1 - x0, y1 - y0) * 1.5 + 60
        self.canvas.set_image(img, (x0 - m, y0 - m, x1 + m, y1 + m))
        self.focus_set()
        self._status()

    def _status(self):
        s = self.sel
        cur = ""
        if s is not None:
            b = self.boxes[s]
            cur = f"    selected: {b['cls']} {b['b'][2] - b['b'][0]:.0f}×{b['b'][3] - b['b'][1]:.0f} px"
        self.top.configure(text=f"RE-ANNOTATE  {self.title}    boxes: {len(self.boxes)}    new boxes get: {self.default}"
                                f"{cur}" + (f"    ·    {self.msg}" if self.msg else ""))
        self.legend.configure(text="Class:  " + "    ".join(f"{k}={n}" for n, k in self.keys.items()))

    def _update(self):
        self._status()
        self.canvas.draw_overlay()

    # ---- drawing
    def _overlay(self, c):
        c.box(self.orig, "#ffe040", 2, dash=(6, 4))
        c.text(self.orig[0], self.orig[3], "original", "#ffe040", anchor="nw", size=12)
        for n in self.neigh:
            c.box(n, "#7a7a7a", 1)
        names = list(self.keys)
        for i, b in enumerate(self.boxes):
            col = class_color(b["cls"], names)
            c.box(b["b"], col, 4 if i == self.sel else 2)
            c.text(b["b"][0], b["b"][1], b["cls"], col, size=14)
        if self.sel is not None:
            for p in handle_points(self.boxes[self.sel]["b"]).values():
                cx, cy = c.to_canvas(*p)
                c.create_rectangle(cx - S(4), cy - S(4), cx + S(4), cy + S(4), fill="#fff", outline="#000", tags="ov")
        d = self.drag_state
        if d and d["mode"] == "new":
            c.box(norm([*d["p0"], *d["p1"]]), "#ffffff", 2, dash=(4, 3))

    # ---- mouse
    def _clamp(self, p):
        ih, iw = self.img.shape[:2]
        return min(max(p[0], 0), iw), min(max(p[1], 0), ih)

    def _handle_at(self, b, cx, cy):
        for name, p in handle_points(b).items():
            hx, hy = self.canvas.to_canvas(*p)
            if abs(hx - cx) <= S(HANDLE_PX) and abs(hy - cy) <= S(HANDLE_PX):
                return name
        return None

    def _press(self, e):
        self.focus_set()
        p = self._clamp(self.canvas.to_img(e.x, e.y))
        if self.sel is not None:
            h = self._handle_at(self.boxes[self.sel]["b"], e.x, e.y)
            if h:
                self.drag_state = {"mode": "resize", "h": h, "i": self.sel}
                return
        i = pick_box([b["b"] for b in self.boxes], p, self.sel)
        if i is not None:
            self.sel = i
            self.drag_state = {"mode": "move", "i": i, "p0": p, "b0": list(self.boxes[i]["b"])}
        else:
            self.sel = None
            self.drag_state = {"mode": "new", "p0": p, "p1": p, "c0": (e.x, e.y)}
        self.msg = ""
        self._update()

    def _drag(self, e):
        d = self.drag_state
        if not d:
            return
        p = self._clamp(self.canvas.to_img(e.x, e.y))
        if d["mode"] == "new":
            d["p1"] = p
        elif d["mode"] == "resize":
            self.boxes[d["i"]]["b"] = resize_box(self.boxes[d["i"]]["b"], d["h"], p)
        else:
            ih, iw = self.img.shape[:2]
            b0 = d["b0"]
            dx = min(max(p[0] - d["p0"][0], -b0[0]), iw - b0[2])
            dy = min(max(p[1] - d["p0"][1], -b0[1]), ih - b0[3])
            self.boxes[d["i"]]["b"] = [b0[0] + dx, b0[1] + dy, b0[2] + dx, b0[3] + dy]
        self._update()

    def _release(self, e):
        d, self.drag_state = self.drag_state, None
        if not d:
            return
        if d["mode"] == "new":
            far = abs(e.x - d["c0"][0]) >= MIN_DRAG_PX and abs(e.y - d["c0"][1]) >= MIN_DRAG_PX
            b = norm([*d["p0"], *d["p1"]])
            if far and b[2] - b[0] >= MIN_BOX and b[3] - b[1] >= MIN_BOX:
                self.boxes.append({"b": b, "cls": self.default})
                self.sel = len(self.boxes) - 1
        else:
            self.boxes[d["i"]]["b"] = norm(self.boxes[d["i"]]["b"])
        self._update()

    # ---- keys
    def result(self):
        ih, iw = self.img.shape[:2]
        out = []
        for b in self.boxes:
            x0, y0 = max(0, round(b["b"][0])), max(0, round(b["b"][1]))
            x1, y1 = min(iw, round(b["b"][2])), min(ih, round(b["b"][3]))
            if x1 - x0 >= MIN_BOX and y1 - y0 >= MIN_BOX:
                out.append({"bbox": [x0, y0, x1 - x0, y1 - y0], "class": b["cls"]})
        return out

    def on_key(self, e):
        ks, ch = e.keysym, printable(e)
        by_key = {k: n for n, k in self.keys.items()}
        if ks in ("Return", "KP_Enter"):
            self.done(self.result())
        elif ks == "Escape":
            self.done(None)
        elif ks in ("Delete", "BackSpace"):
            if self.sel is not None:
                del self.boxes[self.sel]
                self.sel = min(self.sel, len(self.boxes) - 1) if self.boxes else None
            self._update()
        elif ks in ("Tab", "ISO_Left_Tab", "Shift_Tab"):
            if self.boxes:
                step = 1 if ks == "Tab" else -1
                self.sel = 0 if self.sel is None else (self.sel + step) % len(self.boxes)
            self._update()
        elif ks in ("Left", "Right", "Up", "Down"):
            if self.sel is not None:
                step = 10 if e.state & 0x1 else 1
                dx, dy = {"Left": (-step, 0), "Right": (step, 0), "Up": (0, -step), "Down": (0, step)}[ks]
                b = self.boxes[self.sel]["b"]
                ih, iw = self.img.shape[:2]
                dx = min(max(dx, -b[0]), iw - b[2])
                dy = min(max(dy, -b[1]), ih - b[3])
                self.boxes[self.sel]["b"] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
            self._update()
        elif ks == "r":
            self.boxes = [{"b": list(self.orig), "cls": self.default}]
            self.sel = 0
            self._update()
        elif ks == "n":
            raw = simpledialog.askstring("New class", "Class name:", parent=self)
            if raw:
                name, self.msg = self.new_class(raw)
                if name:
                    self._set_class(name)
            self._update()
        elif by_key.get(ch) or by_key.get(ks):
            self._set_class(by_key.get(ch) or by_key.get(ks))
            self._update()

    def _set_class(self, name):
        self.default = name
        if self.sel is not None:
            self.boxes[self.sel]["cls"] = name
