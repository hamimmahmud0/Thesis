"""Per-image viewer: see every annotation of one image at once, to spot objects that have no box.

Opened from review.py with V (or `--view`). Left: the image with all boxes; right: image list, legend, filters.

Mouse   wheel / middle-drag      zoom (around the cursor) / pan
        click                    select the box under the cursor
        drag on the image        draw a box for a missing object, then press a class key (Esc = discard)
Keys    Left / Right (PgUp/PgDn) previous / next image        Home / End   first / last image
        Tab                      hold the boxes away: toggle box visibility to look at the bare image
        class key                change the selected box's class (or give the box you just drew its class)
        C / Space / Enter        confirm the selected box        X / Delete   reject it (delete it if you added it)
        F                        flag it                         R   re-annotate it in the box editor
        N                        new class                       V / Esc   back to the review queue
Boxes you add are stored in <out>/added.json and exported like confirmed crops.
"""
import tkinter as tk
from tkinter import simpledialog

from boxcanvas import BoxCanvas, S, class_color
from editor import MIN_BOX, MIN_DRAG_PX, norm, pick_box, printable, xywh_to_xyxy

BG, PANEL_BG, FG, DIM = "#1c1c1c", "#242424", "#dddddd", "#8a8a8a"
# status -> (legend text, colour or None = class colour, dash, width, shown by default)
STATUS = {
    "confirmed": ("confirmed", None, None, 2, True),
    "changed": ("class changed", None, None, 2, True),
    "redrawn": ("re-annotated", None, None, 2, True),
    "added": ("added by you", None, None, 3, True),
    "undecided": ("not reviewed yet", "#ffe040", None, 1, True),
    "flagged": ("flagged", "#ff9a2e", (6, 3), 2, True),
    "rejected": ("rejected", "#ff4d4d", (6, 3), 1, True),
    "ignored": ("ignored (category / filters)", "#9a9a9a", (2, 3), 1, True),
    "removed": ("auto-removed pedestrian", "#5f5f5f", (2, 3), 1, False),
}
ACCEPTED = ("confirmed", "changed", "redrawn", "added")
NOT_COUNTED = ("ignored", "removed")


def image_stats(items):
    """(accepted, total reviewed, not yet reviewed) for one image's items."""
    acc = sum(1 for i in items if i["status"] in ACCEPTED)
    todo = sum(1 for i in items if i["status"] == "undecided")
    return acc, sum(1 for i in items if i["status"] not in NOT_COUNTED), todo


class ImageViewer(tk.Frame):
    def __init__(self, master, host):
        """host: images (list of COCO image dicts), keys (live name->shortcut dict), items(image_id) -> item dicts,
        get_img(file) -> BGR array, apply(item, op, cls) -> message, add(image_id, bbox, cls) -> message,
        edit(item) -> message | None, new_class(raw) -> (name | None, message), leave(), quit()."""
        super().__init__(master, bg=BG)
        self.host = host
        self.idx, self.items, self.sel, self.hover, self.pending, self.peek = 0, [], None, None, None, False
        self.status_on = {k: tk.BooleanVar(self, v[4]) for k, v in STATUS.items()}
        self.cls_on, self.show_labels = {}, tk.BooleanVar(self, True)
        self.stats = {}
        self.drag = None

        side = tk.Frame(self, bg=PANEL_BG, width=S(400))
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        self.title = tk.Label(side, bg=PANEL_BG, fg="#fff", anchor="w", justify="left", font=("TkDefaultFont", S(14), "bold"),
                              padx=8, pady=6, wraplength=S(384))
        self.title.pack(fill="x")
        # packed first so the list can never squeeze it out: hover / selection info
        self.info = tk.Label(side, bg="#101010", fg=FG, anchor="nw", justify="left", padx=8, pady=6, wraplength=S(384), height=5)
        self.info.pack(side="bottom", fill="x", padx=6, pady=6)
        nav = tk.Frame(side, bg=PANEL_BG)
        nav.pack(fill="x", padx=6)
        tk.Button(nav, text="◀ Prev", command=lambda: self.step(-1)).pack(side="left")
        tk.Button(nav, text="Next ▶", command=lambda: self.step(1)).pack(side="left", padx=4)
        self.goto = tk.Entry(nav, width=8)
        self.goto.pack(side="right")
        self.goto.bind("<Return>", self._goto)
        tk.Label(nav, text="go to", bg=PANEL_BG, fg=DIM).pack(side="right", padx=4)

        lf = tk.Frame(side, bg=PANEL_BG)
        lf.pack(fill="both", expand=True, padx=6, pady=6)
        sb = tk.Scrollbar(lf)
        sb.pack(side="right", fill="y")
        self.listbox = tk.Listbox(lf, height=6, exportselection=False, bg="#181818", fg=FG, selectbackground="#3b5bdb",
                                  activestyle="none", font=("TkFixedFont", S(12)), yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.configure(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self._on_list)

        self.legend = tk.Frame(side, bg=PANEL_BG)
        self.legend.pack(fill="x", padx=6)
        self.classes = tk.Frame(side, bg=PANEL_BG)
        self.classes.pack(fill="x", padx=6, pady=(6, 0))
        tk.Checkbutton(side, text="show class labels", variable=self.show_labels, command=self._overlay_only, bg=PANEL_BG,
                       fg=FG, selectcolor="#000", activebackground=PANEL_BG, activeforeground=FG, anchor="w").pack(fill="x", padx=6)

        self.canvas = BoxCanvas(self, overlay=self._draw)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.hint = tk.Label(self, bg=BG, fg="#5aff78", anchor="w", justify="left", padx=10, pady=4,
                             text="◀ ▶ image   drag = draw missing box   click = select   class key = set class   C confirm   X reject   "
                                  "F flag   R re-annotate   Tab = hide boxes   N new class   V back   Q quit")
        self.hint.pack(side="bottom", fill="x")
        self.bind("<Configure>", lambda e: self.hint.configure(wraplength=max(200, e.width - S(430))))
        c = self.canvas
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<B1-Motion>", self._move)
        c.bind("<ButtonRelease-1>", self._release)
        c.bind("<Motion>", self._hover)

    # ---- loading
    def open(self, image_id=None, select=None):
        imgs = self.host.images
        self.listbox.delete(0, "end")
        self.stats = {}
        for i, im in enumerate(imgs):
            self.stats[im["id"]] = image_stats(self.host.items(im["id"]))
            self.listbox.insert("end", self._row(i))
        ids = [im["id"] for im in imgs]
        self.load(ids.index(image_id) if image_id in ids else self.idx, select)

    def _row(self, i):
        im = self.host.images[i]
        acc, tot, todo = self.stats[im["id"]]
        return f"{i + 1:>3} {im['file_name'][-18:]:<18} {acc:>3}/{tot:<3}" + (f" ?{todo}" if todo else "")

    def load(self, idx, select=None, zoom=True):
        n = len(self.host.images)
        if not n:
            return
        self.idx = max(0, min(n - 1, idx))
        im = self.host.images[self.idx]
        self.items = self.host.items(im["id"])
        self.sel = select if any(i["id"] == select for i in self.items) else None
        self.pending = self.hover = None
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(self.idx)
        self.listbox.see(self.idx)
        rect, margin = None, 0.0
        it = self._item(self.sel)
        if it and zoom:
            x, y, w, h = it["bbox"]
            m = max(w, h) * 2 + 80
            rect = (x - m, y - m, x + w + m, y + h + m)
        self.canvas.set_image(self.host.get_img(im["file_name"]), rect, margin)
        self._sidebar()
        self.canvas.focus_set()

    def reload(self):
        """Items changed (a decision was made): rebuild the lists but keep the view."""
        im = self.host.images[self.idx]
        self.items = self.host.items(im["id"])
        if not self._item(self.sel):
            self.sel = None
        self.stats[im["id"]] = image_stats(self.items)
        self.listbox.delete(self.idx)
        self.listbox.insert(self.idx, self._row(self.idx))
        self.listbox.selection_set(self.idx)
        self._sidebar()
        self.canvas.draw_overlay()

    def step(self, d):
        self.load(self.idx + d, zoom=False)

    def _goto(self, _e=None):
        q = self.goto.get().strip()
        imgs = self.host.images
        if q.isdigit() and 1 <= int(q) <= len(imgs):
            self.load(int(q) - 1, zoom=False)
        elif q:
            hit = next((i for i, im in enumerate(imgs) if q.lower() in im["file_name"].lower()), None)
            if hit is not None:
                self.load(hit, zoom=False)
        self.canvas.focus_set()

    def _on_list(self, _e):
        sel = self.listbox.curselection()
        if sel and sel[0] != self.idx:
            self.load(sel[0], zoom=False)
        self.canvas.focus_set()

    # ---- sidebar
    def _sidebar(self):
        im = self.host.images[self.idx]
        acc, tot, todo = self.stats[im["id"]]
        self.title.configure(text=f"Image {self.idx + 1} / {len(self.host.images)}\n{im['file_name']}\n"
                                  f"{acc} accepted · {todo} not reviewed · {tot} total")
        for w in self.legend.winfo_children():
            w.destroy()
        count = {}
        for i in self.items:
            count[i["status"]] = count.get(i["status"], 0) + 1
        for k, (label, color, *_rest) in STATUS.items():
            if not count.get(k):   # only statuses present in this image; keeps the sidebar short
                continue
            row = tk.Frame(self.legend, bg=PANEL_BG)
            row.pack(fill="x")
            tk.Label(row, bg=color or "#4cc9f0", width=2).pack(side="left", padx=(0, 4))
            tk.Checkbutton(row, text=f"{label} ({count.get(k, 0)})", variable=self.status_on[k], command=self._overlay_only,
                           bg=PANEL_BG, fg=FG if count.get(k) else DIM, selectcolor="#000", activebackground=PANEL_BG,
                           activeforeground=FG, anchor="w", pady=0).pack(side="left", fill="x")
        for w in self.classes.winfo_children():
            w.destroy()
        per = {}
        for i in self.items:
            if i["status"] in ACCEPTED or i["status"] == "undecided":
                per[i["cls"]] = per.get(i["cls"], 0) + 1
        names = list(self.host.keys)
        for n in sorted(per, key=lambda n: names.index(n) if n in names else 99):
            var = self.cls_on.setdefault(n, tk.BooleanVar(self, True))
            tk.Checkbutton(self.classes, text=f"{n} ({per[n]})", variable=var, command=self._overlay_only, bg=PANEL_BG,
                           fg=class_color(n, names), selectcolor="#000", activebackground=PANEL_BG,
                           activeforeground=FG, anchor="w").grid(row=len(self.classes.grid_slaves()) // 2,
                                                                 column=len(self.classes.grid_slaves()) % 2, sticky="w")
        self._info()

    def _describe(self, it):
        x, y, w, h = it["bbox"]
        s = f"{STATUS[it['status']][0]}: {it['cls']}\n{int(w)}×{int(h)} px"
        if it.get("score") is not None:
            s += f"   score {it['score']:.2f}"
        return s + f"\nid {it['id']}"

    def _info(self, msg=""):
        sel, hov = self._item(self.sel), self._item(self.hover)
        lines = []
        if sel:
            lines.append("SELECTED  " + self._describe(sel).replace("\n", "  ·  ", 1))
        if hov and hov is not sel:
            lines.append("hover  " + self._describe(hov).replace("\n", "  ·  ", 1))
        if self.pending:
            lines.append("New box: press a class key (Esc = discard)")
        if not lines and not msg:
            lines.append("Click a box to select it.\nDrag on the image to draw a missing object.")
        self.info.configure(text="\n".join(lines + ([msg] if msg else [])), fg="#ff9a9a" if msg else FG)

    # ---- drawing
    def _item(self, item_id):
        return next((i for i in self.items if i["id"] == item_id), None) if item_id is not None else None

    def _visible(self, it):
        return self.status_on[it["status"]].get() and (
            it["status"] not in ACCEPTED + ("undecided",) or self.cls_on.get(it["cls"], tk.BooleanVar(value=True)).get())

    def _overlay_only(self):
        self.canvas.draw_overlay()

    def _draw(self, c):
        if self.peek:
            return
        names = list(self.host.keys)
        show = [i for i in self.items if self._visible(i)]
        for it in sorted(show, key=lambda i: -i["bbox"][2] * i["bbox"][3]):  # small boxes end up on top
            _, color, dash, width, _ = STATUS[it["status"]]
            color = color or class_color(it["cls"], names)
            b = xywh_to_xyxy(it["bbox"])
            c.box(b, color, width, dash)
            (x0, _), (x1, _) = c.to_canvas(b[0], b[1]), c.to_canvas(b[2], b[3])
            if self.show_labels.get() and it["status"] not in NOT_COUNTED and x1 - x0 >= S(30):
                c.text(b[0], b[1], it["cls"], color, size=13)
        it = self._item(self.sel)
        if it:
            c.box(xywh_to_xyxy(it["bbox"]), "#ffffff", 4)
        if self.pending:
            c.box(self.pending, "#ffffff", 2, dash=(4, 3))
            c.text(self.pending[0], self.pending[1], "? press a class key", "#ffffff", size=13)
        d = self.drag
        if d and d["moved"]:
            c.box(norm([*d["p0"], *d["p1"]]), "#ffffff", 2, dash=(4, 3))

    # ---- mouse
    def _hit(self, p):
        vis = [i for i in self.items if self._visible(i)]
        k = pick_box([xywh_to_xyxy(i["bbox"]) for i in vis], p)
        return vis[k]["id"] if k is not None else None

    def _hover(self, e):
        if self.peek:
            return
        h = self._hit(self.canvas.to_img(e.x, e.y))
        if h != self.hover:
            self.hover = h
            self._info()

    def _press(self, e):
        self.canvas.focus_set()
        p = self.canvas.to_img(e.x, e.y)
        self.drag = {"p0": p, "p1": p, "c0": (e.x, e.y), "moved": False}

    def _move(self, e):
        d = self.drag
        if d:
            d["p1"] = self.canvas.to_img(e.x, e.y)
            d["moved"] = d["moved"] or max(abs(e.x - d["c0"][0]), abs(e.y - d["c0"][1])) >= MIN_DRAG_PX
            self.canvas.draw_overlay()

    def _release(self, e):
        d, self.drag = self.drag, None
        if not d:
            return
        if d["moved"]:
            ih, iw = self.canvas.img.shape[:2]
            b = norm([*d["p0"], *d["p1"]])
            b = [min(max(b[0], 0), iw), min(max(b[1], 0), ih), min(max(b[2], 0), iw), min(max(b[3], 0), ih)]
            self.pending = b if b[2] - b[0] >= MIN_BOX and b[3] - b[1] >= MIN_BOX else None
            self.sel = None
        else:
            self.pending = None
            self.sel = self._hit(d["p0"])
        self.canvas.draw_overlay()
        self._info()

    # ---- keys
    def on_key(self, e):
        ks, ch = e.keysym, printable(e)
        by_key = {k: n for n, k in self.host.keys.items()}
        cls = by_key.get(ch) or by_key.get(ks)
        it = self._item(self.sel)
        if ks in ("Left", "Prior", "Up"):
            self.step(-1)
        elif ks in ("Right", "Next", "Down"):
            self.step(1)
        elif ks == "Home":
            self.load(0, zoom=False)
        elif ks == "End":
            self.load(len(self.host.images) - 1, zoom=False)
        elif ks == "Tab":
            self.peek = not self.peek
            self.canvas.draw_overlay()
        elif ks in ("v", "V") or (ks == "Escape" and not self.pending):
            self.host.leave()
        elif ks == "Escape":
            self.pending = None
            self.canvas.draw_overlay()
            self._info()
        elif ks == "q":
            self.host.quit()
        elif ks == "n":
            raw = simpledialog.askstring("New class", "Class name:", parent=self)
            if raw:
                name, msg = self.host.new_class(raw)
                if name:
                    self._sidebar()
                    self._info(f"class {name} created, key {self.host.keys[name]}")
                else:
                    self._info(msg)
        elif cls and self.pending:
            im = self.host.images[self.idx]
            msg = self.host.add(im["id"], self.pending, cls)
            self.pending = None
            self.reload()
            self._info(msg)
        elif ks in ("space", "Return", "KP_Enter", "c", "x", "Delete", "f", "r") or cls:
            if not it:
                self._info("select a box first (click it)")
                return
            op = {"space": "confirm", "Return": "confirm", "KP_Enter": "confirm", "c": "confirm", "x": "reject",
                  "Delete": "reject", "f": "flag", "r": "edit"}.get(ks, "class")
            if op == "edit":
                msg = self.host.edit(it)
            else:
                msg = self.host.apply(it, op, cls)
                self.reload()
            self._info(msg or "")
