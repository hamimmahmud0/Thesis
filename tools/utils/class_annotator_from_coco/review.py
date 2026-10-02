#!/usr/bin/env python3
"""Review COCO detections one object at a time and build a classification dataset.

Keys
  Space / Enter / C   confirm (keep predicted class)
  1..8                change class to that category id and confirm
  X / Delete          reject object from the dataset
  Backspace / U       undo last decision
  S                   skip for now (decide later)
  F                   flag for later inspection (several objects in one box, bad box, ...)
  Tab / Shift+Tab     jump to next / previous class
  H                   toggle HISTORY: browse decided objects (<- older, -> newer) and fix them
                      with the same keys (confirm / class key / X / F / N)
  Q / Esc             save and quit

Decisions are saved to <out>/decisions.json after every key press, so you can
quit and resume at any time. Crops are written to <out>/<class_name>/ on exit
(and with --export-only).

Usage:
  python review.py                       # review dataset/ -> cls_dataset/
  python review.py --min-score 0.5 --min-size 24
  python review.py --max-per-class 200   # at most 200 accepted per class
  python review.py --classes my_classes.yaml   # other default classes (see classes.yaml)
  python review.py --history             # start in history mode to recheck earlier decisions
  python review.py --export-only
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

WIN = "review"
PANEL = 720          # size of each of the two panels (px); shrunk to fit the screen in main()
# class shortcuts: digits, then letters (minus the command keys), then Shift+letters
RESERVED = "cxsuqnfh"   # command keys, never used as class shortcuts
KEY_POOL = "1234567890" + "".join(c for c in "abcdefghijklmnopqrstuvwxyz" if c not in RESERVED) + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NEW_ID = 1000        # ids for classes added during annotation start here
PAD = 0.25           # context padding around bbox for the crop panel


def load_classes(path, categories):
    """Default classes from YAML -> (cats id->name, predicted ids, keys name->key or None)."""
    import yaml
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    coco = {c["name"]: c["id"] for c in categories}
    cats, keys = {}, {}
    for item in cfg.get("classes", []):
        item = {"name": item} if isinstance(item, str) else item
        name = str(item["name"])
        key = item.get("key")
        key = None if key is None else str(key)
        if name in keys:
            sys.exit(f"{path}: duplicate class {name!r}")
        if key is not None and (len(key) != 1 or key in RESERVED or key in keys.values()):
            sys.exit(f"{path}: bad or duplicate key {key!r} for {name!r}")
        cats[coco.get(name, NEW_ID + len(cats))] = name
        keys[name] = key
    if not cats:
        sys.exit(f"{path}: no classes defined")
    return cats, {n for n in cats.values() if n not in coco}, keys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--out", default="cls_dataset")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--min-size", type=int, default=0, help="skip boxes with min(w,h) below this")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="stop showing a class once this many objects are accepted for it (0 = no cap); "
                        "also shuffles image order so picks spread across frames")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed used with --max-per-class")
    p.add_argument("--classes", default=str(Path(__file__).with_name("classes.yaml")),
                   help="YAML file with the default classes (see classes.yaml)")
    p.add_argument("--class-order", nargs="*", default=[], metavar="CLASS",
                   help="review predicted classes in this order (default: category order in the COCO file)")
    p.add_argument("--only-flagged", action="store_true",
                   help="re-review just the objects flagged earlier with F (new decisions replace the flag)")
    p.add_argument("--history", action="store_true", help="start in history mode (recheck / edit earlier decisions)")
    p.add_argument("--export-only", action="store_true")
    p.add_argument("--pad", type=int, default=8, help="extra pixels around bbox in exported crops")
    return p.parse_args()


def load_state(out: Path):
    f = out / "decisions.json"
    if f.exists():
        return json.loads(f.read_text())
    return {}


def save_state(out: Path, dec):
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "decisions.json.tmp"
    tmp.write_text(json.dumps(dec))
    tmp.replace(out / "decisions.json")


def crop_box(img, bbox, pad_frac, pad_px=0):
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    px = max(int(bw * pad_frac), pad_px)
    py = max(int(bh * pad_frac), pad_px)
    x0, y0 = max(0, int(x) - px), max(0, int(y) - py)
    x1, y1 = min(w, int(x + bw) + px), min(h, int(y + bh) + py)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def fit(img, size):
    h, w = img.shape[:2]
    s = size / max(h, w)
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=interp)
    canvas = np.zeros((size, size, 3), np.uint8)
    oy, ox = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    return canvas


def render(img, ann):
    x, y, bw, bh = ann["bbox"]
    # left: tight crop; right: wide context with box drawn
    tight, _ = crop_box(img, ann["bbox"], 0.10)
    left = fit(tight, PANEL)
    cx, cy = x + bw / 2, y + bh / 2
    half = int(max(bw, bh) * 3 + 100)
    ctx_box = [cx - half, cy - half, 2 * half, 2 * half]
    ctx, (x0, y0, x1, y1) = crop_box(img, ctx_box, 0)
    ctx = ctx.copy()
    cv2.rectangle(ctx, (int(x - x0), int(y - y0)), (int(x + bw - x0), int(y + bh - y0)), (0, 255, 255), max(2, half // 100))
    right = fit(ctx, PANEL)

    return np.hstack([left, right])


def ui_font(size, bold=False):
    from PIL import ImageFont
    d = "/usr/share/fonts/truetype/ubuntu/"
    for p in (d + ("Ubuntu-B.ttf" if bold else "Ubuntu-R.ttf"), "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def wrap(d, items, font, width):
    lines, line = [], ""
    for it in items:
        cand = f"{line}    {it}" if line else it
        if line and d.textlength(cand, font=font) > width:
            lines.append(line)
            line = it
        else:
            line = cand
    return lines + [line]


def compose(img, ann, cats, keys, cur, cls_prog, total, n_done, counts=None, cap=0, msg="", prompt=None, hist=None):
    """Image panels plus Ubuntu-font text, as a PIL image."""
    from PIL import Image, ImageDraw
    panels = Image.fromarray(cv2.cvtColor(render(img, ann), cv2.COLOR_BGR2RGB))
    W = panels.width
    big, mid, small, tiny = ui_font(34, True), ui_font(22), ui_font(24), ui_font(20)
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    cls_lines = wrap(d, [f"{k}={n}" for n, k in keys.items()], small, W - 200)
    acc_lines = wrap(d, [f"{n} {counts.get(n, 0)}/{cap}" for n in keys], tiny, W - 140) if cap else []
    top_h = 96
    bot_h = 20 + 40 + 34 * len(cls_lines) + (8 + 28 * len(acc_lines) if cap else 0) + 14
    canvas = Image.new("RGB", (W, top_h + panels.height + bot_h), (0, 0, 0))
    canvas.paste(panels, (0, top_h))
    d = ImageDraw.Draw(canvas)
    if hist:
        d.rectangle([0, 0, W, top_h - 1], fill=(52, 28, 92))
    bw, bh = ann["bbox"][2:]
    name = cats[ann["category_id"]]
    d.text((16, 8), name, font=big, fill=(255, 255, 255))
    x = 16 + d.textlength(f"{name}   ", font=big)
    d.text((x, 14), f"score {ann.get('score', 0):.2f}  ·  {int(bw)}×{int(bh)} px", font=mid, fill=(200, 200, 200))
    if cur:
        txt = {"reject": "rejected", "flag": "flagged"}.get(cur["action"]) or \
            f"{'confirmed' if cur['action'] == 'confirm' else 'changed to'} {cur['class']}"
        d.text((W - 16 - d.textlength(txt, font=mid), 14), txt, font=mid, fill=(255, 190, 80))
    if hist:
        d.text((16, 58), f"HISTORY {hist[0]} / {hist[1]}  (1 = oldest)    {ann['file']}", font=tiny, fill=(215, 190, 255))
        hint = "LEFT / RIGHT  older / newer    H  back to queue    SPACE  confirm    X  reject    F  flag    N  new class    Q  quit"
    else:
        d.text((16, 58), f"{name} {cls_prog[0]} / {cls_prog[1]}    all {n_done} / {total}    {ann['file']}", font=tiny, fill=(150, 150, 150))
        hint = "SPACE/ENTER  confirm    X  reject    F  flag    S  skip    U  undo    TAB  next class    N  new class    H  history    Q  quit"
    y = top_h + panels.height + 10
    d.text((16, y), hint, font=small, fill=(90, 255, 120))
    y += 40
    d.text((16, y), "Class:", font=small, fill=(255, 200, 100))
    for ln in cls_lines:
        d.text((110, y), ln, font=small, fill=(255, 200, 100))
        y += 34
    if cap:
        y += 8
        d.text((16, y), "Accepted:", font=tiny, fill=(160, 200, 255))
        for ln in acc_lines:
            d.text((110, y), ln, font=tiny, fill=(160, 200, 255))
            y += 28
    if prompt is not None:
        d.rectangle([0, 0, W, 54], fill=(30, 30, 90))
        d.text((16, 12), f"New class name: {prompt}|", font=mid, fill=(255, 255, 255))
        hint = "Enter = add & apply      Esc = cancel"
        d.text((W - 16 - d.textlength(hint, font=tiny), 16), hint, font=tiny, fill=(200, 200, 255))
    elif msg:
        d.text((W - 16 - d.textlength(msg, font=mid), 58), msg, font=mid, fill=(255, 90, 90))
    return canvas


def flagged_dir(args):
    return Path(str(Path(args.out).resolve()) + "_flagged")


def export(args, data, cats, dec):
    out = Path(args.out)
    for d in out.iterdir() if out.exists() else []:
        if d.is_dir():
            shutil.rmtree(d)
    fdir = flagged_dir(args)
    for d in fdir.iterdir() if fdir.exists() else []:
        if d.is_dir():
            shutil.rmtree(d)
    imgs = {i["id"]: i for i in data["images"]}
    anns = {str(a["id"]): a for a in data["annotations"]}
    by_img = {}
    for aid, d in dec.items():
        if d["action"] in ("confirm", "change"):
            by_img.setdefault(anns[aid]["image_id"], []).append((aid, d))
    flagged = [(aid, anns[aid]) for aid, d in dec.items() if d["action"] == "flag"]
    for aid, a in flagged:  # flagged objects are cropped with extra context so a bad box is easy to see
        by_img.setdefault(a["image_id"], [])
    n = 0
    for iid, items in by_img.items():
        im = cv2.imread(str(Path(args.dataset) / "images" / imgs[iid]["file_name"]))
        stem = Path(imgs[iid]["file_name"]).stem
        for aid, d in items:
            crop, _ = crop_box(im, anns[aid]["bbox"], 0, args.pad)
            cdir = out / d["class"]
            cdir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cdir / f"{stem}_{aid}.png"), crop)
            n += 1
        for aid, a in flagged:
            if a["image_id"] == iid:
                crop, _ = crop_box(im, a["bbox"], 0.5, args.pad)
                cdir = fdir / cats[a["category_id"]]
                cdir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cdir / f"{stem}_{aid}.png"), crop)
    if not flagged:
        (fdir / "flagged.json").unlink(missing_ok=True)
    else:
        fdir.mkdir(parents=True, exist_ok=True)
        (fdir / "flagged.json").write_text(json.dumps([
            {"annotation_id": a["id"], "image": imgs[a["image_id"]]["file_name"], "bbox": a["bbox"],
             "predicted": cats[a["category_id"]], "score": a.get("score")} for _, a in flagged], indent=1))
    print(f"exported {n} crops to {out}/" + (f", {len(flagged)} flagged to {fdir}/" if flagged else ""))


def main():
    args = parse_args()
    ds = Path(args.dataset)
    data = json.loads((ds / "annotations" / "instances.json").read_text())
    cats, label_only, keys = load_classes(args.classes, data["categories"])
    imgs = {i["id"]: i for i in data["images"]}
    out = Path(args.out)
    dec = load_state(out)

    anns = []
    for a in sorted(data["annotations"], key=lambda a: (a["image_id"], a["id"])):
        if a["category_id"] not in cats or a.get("score", 1.0) < args.min_score or min(a["bbox"][2:]) < args.min_size:
            continue
        a["file"] = imgs[a["image_id"]]["file_name"]
        anns.append(a)
    # review one predicted class at a time; within a class keep objects of the same image together
    order = sorted({a["image_id"] for a in anns})
    if args.max_per_class:  # with a cap, spread picks across frames instead of taking the first ones
        import random
        random.Random(args.seed).shuffle(order)
    rank = {iid: r for r, iid in enumerate(order)}
    names = list(cats.values())
    first = [n for n in args.class_order if n in names]
    class_rank = {n: r for r, n in enumerate(first + [n for n in names if n not in first])}
    anns.sort(key=lambda a: (class_rank[cats[a["category_id"]]], rank[a["image_id"]], a["id"]))

    if args.export_only:
        export(args, data, cats, dec)
        return

    # extra classes added during earlier sessions: [{"name":..., "key":...}]
    extra_file = out / "classes.json"
    extra = json.loads(extra_file.read_text()) if extra_file.exists() else []
    extra = [e for e in extra if e["name"] not in keys]  # now defined in classes.yaml
    for e in extra:
        if e["key"] in keys.values():  # saved key now belongs to a yaml class: yaml wins, move this one
            e["key"] = next(k for k in KEY_POOL if k not in keys.values() and k not in {x["key"] for x in extra})
            extra_file.write_text(json.dumps(extra))
            print(f"class {e['name']!r} moved to key {e['key']!r} (its old key is used in classes.yaml)")
        cats[NEW_ID + len(cats)] = e["name"]
        keys[e["name"]] = e["key"]
    for n in list(keys):  # classes without a key in the yaml get the next free one
        if keys[n] is None:
            keys[n] = next(k for k in KEY_POOL if k not in keys.values())

    def is_open(a):  # still needs a decision (in --only-flagged mode: still carries the flag)
        d = dec.get(str(a["id"]))
        return d is not None and d["action"] == "flag" if args.only_flagged else d is None

    pending = [i for i, a in enumerate(anns) if is_open(a)]
    ann_idx = {str(a["id"]): i for i, a in enumerate(anns)}
    if not pending and not any(k in ann_idx for k in dec):
        print("nothing left to review")
        export(args, data, cats, dec)
        return

    import tkinter as tk
    from PIL import Image, ImageTk

    skipped = set()
    history = []  # annotation ids in order of decision, for undo
    cache = {"file": None, "img": None}
    st = {"pos": 0, "cur": None, "msg": "", "prompt": None, "hist": None}  # st["hist"]: annotation id shown in history mode
    cap = args.max_per_class

    def counts():
        c = {}
        for d in dec.values():
            if d["action"] in ("confirm", "change"):
                c[d["class"]] = c.get(d["class"], 0) + 1
        return c

    def full(name, c):
        return bool(cap) and c.get(name, 0) >= cap

    cls_total = {}
    for a in anns:
        cls_total[a["category_id"]] = cls_total.get(a["category_id"], 0) + 1

    def cls_progress(a):
        done = sum(1 for b in anns if b["category_id"] == a["category_id"] and str(b["id"]) in dec)
        return min(done + 1, cls_total[a["category_id"]]), cls_total[a["category_id"]]

    def get_img(f):
        if cache["file"] != f:
            cache["img"] = cv2.imread(str(ds / "images" / f))
            cache["file"] = f
        return cache["img"]

    root = tk.Tk()
    root.title(WIN)
    global PANEL
    PANEL = max(300, min(PANEL, (root.winfo_screenheight() - 420), (root.winfo_screenwidth() - 60) // 2))
    root.configure(bg="black")
    label = tk.Label(root, bd=0, bg="black")
    label.pack()

    def finish():
        save_state(out, dec)
        root.destroy()

    def hist_ids():  # decided objects, oldest first
        return [k for k in dec if k in ann_idx]

    def enter_history(aid=None, msg=""):
        ids = hist_ids()
        if not ids:
            st["msg"] = "no decisions yet"
            return False
        st["hist"] = aid if aid in ann_idx and aid in dec else ids[-1]
        st["msg"] = msg
        return True

    def show_history():
        ids = hist_ids()
        if st["hist"] not in dec or st["hist"] not in ann_idx:
            st["hist"] = ids[-1] if ids else None
        if st["hist"] is None:
            show()
            return
        i = ann_idx[st["hist"]]
        st["cur"] = i
        a = anns[i]
        photo = ImageTk.PhotoImage(compose(get_img(a["file"]), a, cats, keys, dec.get(st["hist"]), None, len(anns), len(dec),
                                           counts(), cap, st["msg"], st["prompt"], (ids.index(st["hist"]) + 1, len(ids))))
        st["msg"] = ""
        label.configure(image=photo)
        label.image = photo

    def show():
        if st["hist"] is not None:
            show_history()
            return
        c = counts()

        def todo(p):
            j = pending[p]
            return not (not is_open(anns[j]) or j in skipped or full(cats[anns[j]["category_id"]], c))

        # next open item at/after pos, wrapping to the start (classes may have been left half-done via Tab)
        pos = next((p for p in list(range(st["pos"], len(pending))) + list(range(0, st["pos"])) if todo(p)), None)
        if pos is None:
            if enter_history(msg="queue finished: history mode, Q to quit"):
                show_history()
            else:
                finish()
            return
        st["pos"] = pos
        i = pending[pos]
        st["cur"] = i
        a = anns[i]
        photo = ImageTk.PhotoImage(compose(get_img(a["file"]), a, cats, keys, dec.get(str(a["id"])), cls_progress(a), len(anns), len(dec), c, cap, st["msg"], st["prompt"]))
        st["msg"] = ""
        label.configure(image=photo)
        label.image = photo

    def decide(d):
        aid = str(anns[st["cur"]]["id"])
        if st["hist"] is not None:  # editing an earlier decision: overwrite in place, stay on this object
            history[:] = [h for h in history if h[0] != aid]  # keep the undo stack consistent
            dec[aid] = d
            save_state(out, dec)
            st["msg"] = ""
            show()
            return
        history.append((aid, dec.get(aid)))
        dec[aid] = d
        save_state(out, dec)
        st["pos"] += 1
        show()

    def accept(name):
        a = anns[st["cur"]]
        own = dec.get(str(a["id"]), {})
        own = own.get("class") if own.get("action") in ("confirm", "change") else None
        if name != own and full(name, counts()):
            st["msg"] = f"{name} is full"
            show()
            return
        decide({"action": "confirm" if name == cats[a["category_id"]] else "change", "class": name})

    def add_class(raw):
        name = "_".join(raw.split()).replace("/", "-").replace("\\", "-")
        if not name or name.startswith("."):
            st["msg"] = "invalid name"
        elif name.lower() in (n.lower() for n in keys):
            st["msg"] = f"{name} already exists"
        else:
            free = next((k for k in KEY_POOL if k not in keys.values()), None)
            if free is None:
                st["msg"] = "no free shortcut keys left"
            else:
                cats[NEW_ID + len(cats)] = name
                keys[name] = free
                extra.append({"name": name, "key": free})
                out.mkdir(parents=True, exist_ok=True)
                extra_file.write_text(json.dumps(extra))
                accept(name)
                return
        show()

    def switch_class(delta):
        c, firsts = counts(), {}
        for p, j in enumerate(pending):
            n = cats[anns[j]["category_id"]]
            if n not in firsts and is_open(anns[j]) and j not in skipped and not full(n, c):
                firsts[n] = p
        order = sorted(firsts, key=class_rank.get)
        if len(order) < 2:
            st["msg"] = "no other class left to label"
        else:
            cur = cats[anns[st["cur"]]["category_id"]]
            st["pos"] = firsts[order[(order.index(cur) + delta) % len(order)]]
        show()

    def on_key(e):
        ks = e.keysym
        if st["prompt"] is not None:  # typing a new class name
            if ks in ("Return", "KP_Enter"):
                text, st["prompt"] = st["prompt"], None
                add_class(text)
                return
            if ks == "Escape":
                st["prompt"] = None
            elif ks == "BackSpace":
                st["prompt"] = st["prompt"][:-1]
            elif e.char and e.char.isprintable():
                st["prompt"] += e.char
            show()
            return
        a = anns[st["cur"]]
        by_key = {k: n for n, k in keys.items()}
        if ks == "h":
            if st["hist"] is not None:  # back to the queue
                st["hist"] = None
            elif not enter_history(str(a["id"]) if str(a["id"]) in dec else None):
                pass
            show()
            return
        if st["hist"] is not None:
            if ks in ("Left", "Right", "Home", "End"):
                ids = hist_ids()
                k = ids.index(st["hist"])
                k = {"Left": k - 1, "Right": k + 1, "Home": 0, "End": len(ids) - 1}[ks]
                if 0 <= k < len(ids):
                    st["hist"] = ids[k]
                else:
                    st["msg"] = "oldest decision" if k < 0 else "newest decision"
                show()
                return
            if ks in ("Tab", "ISO_Left_Tab", "Shift_Tab", "s", "u", "BackSpace"):
                st["msg"] = "not available in history"
                show()
                return
        if ks in ("space", "Return", "KP_Enter", "c"):
            decide({"action": "confirm", "class": cats[a["category_id"]]})
        elif ks == "Tab":
            switch_class(1)
        elif ks in ("ISO_Left_Tab", "Shift_Tab"):
            switch_class(-1)
        elif ks == "n":
            st["prompt"] = ""
            show()
        elif ks in by_key:
            accept(by_key[ks])
        elif ks == "f":
            decide({"action": "flag"})
        elif ks in ("x", "Delete"):
            decide({"action": "reject"})
        elif ks == "s":
            skipped.add(st["cur"])
            show()
        elif ks in ("u", "BackSpace"):
            if history:
                last, prev = history.pop()
                dec.pop(last, None)
                if prev is not None:
                    dec[last] = prev
                st["pos"] = next(p for p, j in enumerate(pending) if str(anns[j]["id"]) == last)
                save_state(out, dec)
                show()
        elif ks in ("q", "Escape"):
            finish()

    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW", finish)
    if args.history or not pending:
        enter_history(msg="" if args.history else "queue is empty: history mode, Q to quit")
    show()
    try:
        root.mainloop()
    except tk.TclError:  # everything already capped: window closed before the loop started
        pass

    left = sum(1 for a in anns if str(a["id"]) not in dec)
    print(f"decided {len(dec)}, remaining {left}")
    export(args, data, cats, dec)


if __name__ == "__main__":
    sys.exit(main())
