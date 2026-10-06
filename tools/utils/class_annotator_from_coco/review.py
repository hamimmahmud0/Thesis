#!/usr/bin/env python3
"""Review COCO detections one object at a time and build a classification dataset.

Keys
  Space / Enter / C   confirm (keep predicted class)
  <class key>         change class to that class and confirm (keys are listed in the footer / classes.yaml)
  X / Delete          reject object from the dataset
  Backspace / U       undo last decision
  S                   skip for now (decide later)
  F                   flag for later inspection (several objects in one box, bad box, ...)
  R                   RE-ANNOTATE: open the box editor (fix box and class, one or several objects)
  V                   open the per-image VIEWER (all boxes of the image, add missing ones)
  Tab / Shift+Tab     jump to next / previous class
  H                   toggle HISTORY: browse decided objects (<- older, -> newer) and fix them
                      with the same keys; Tab / Shift+Tab there cycles the class filter
  Q / Esc             save and quit

Decisions are saved to <out>/decisions.json after every key press, so you can
quit and resume at any time. Crops are written to <out>/<class_name>/ on exit
(and with --export-only), together with <out>/corrected_instances.json (COCO).
Objects added in the viewer are stored in <out>/added.json.

Usage:
  python review.py                       # review dataset/ -> cls_dataset/
  python review.py --min-score 0.5 --min-size 24
  python review.py --max-per-class 200   # at most 200 accepted per class
  python review.py --classes my_classes.yaml   # other default classes (see classes.yaml)
  python review.py --history             # start in history mode to recheck earlier decisions
  python review.py --recheck Car Bus     # recheck everything decided as Car or Bus
  python review.py --recheck flagged     # ... or the flagged ones (rejected, redrawn work the same way)
  python review.py --only-flagged        # review the flagged detections again (press R to re-annotate)
  python review.py --view                # open the per-image viewer first
  python review.py --export-only
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

WIN = "review"
PANEL = 720          # size of each of the two panels (px); shrunk to fit the screen in main()
# class shortcuts: digits, then letters (minus the command keys), then Shift+letters
RESERVED = "cxsuqnfhrv"   # command keys, never used as class shortcuts
KEY_POOL = "1234567890" + "".join(c for c in "abcdefghijklmnopqrstuvwxyz" if c not in RESERVED) + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NEW_ID = 1000        # ids for classes added during annotation start here
PAD = 0.25           # context padding around bbox for the crop panel
PSEUDO = {"reject": "[rejected]", "flag": "[flagged]", "redraw": "[redrawn]"}   # history filters that are not classes
PSEUDO_ORDER = list(PSEUDO.values())


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
    p.add_argument("--recheck", nargs="+", default=[], metavar="CLASS",
                   help="start in history mode showing only objects decided as these classes "
                        "(or: rejected, flagged, redrawn); Tab in history mode cycles the filter")
    p.add_argument("--view", action="store_true", help="start in the per-image viewer")
    p.add_argument("--ui-scale", type=float, default=1.6,
                   help="size of text and widgets in the box editor and viewer (1 = small, default 1.6)")
    p.add_argument("--export-only", action="store_true")
    p.add_argument("--pad", type=int, default=8, help="extra pixels around bbox in exported crops")
    return p.parse_args()


def load_state(out: Path):
    f = out / "decisions.json"
    if f.exists():
        return json.loads(f.read_text())
    return {}


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def save_state(out: Path, dec):
    save_json(out / "decisions.json", dec)


def accepted_classes(d):
    """Classes an object contributes to the dataset (one per box for a re-annotation)."""
    if d["action"] in ("confirm", "change"):
        return [d["class"]]
    if d["action"] == "redraw":
        return [b["class"] for b in d["boxes"]]
    return []


def decision_tags(d):
    """What a history filter can match: the decided classes plus [rejected] / [flagged] / [redrawn]."""
    tags = set(accepted_classes(d))
    if d["action"] in PSEUDO:
        tags.add(PSEUDO[d["action"]])
    return tags


def decision_text(cur):
    a = cur["action"]
    if a == "reject":
        return "rejected"
    if a == "flag":
        return "flagged"
    if a == "redraw":
        return "re-annotated: " + ", ".join(b["class"] for b in cur["boxes"])
    return f"{'confirmed' if a == 'confirm' else 'changed to'} {cur['class']}"


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


def render(img, ann, extra=None):
    """Tight crop (left) and wide context with the box (right). extra: re-annotated boxes, drawn in magenta."""
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
    for b in extra or []:
        ex, ey, ew, eh = b["bbox"]
        cv2.rectangle(ctx, (int(ex - x0), int(ey - y0)), (int(ex + ew - x0), int(ey + eh - y0)), (255, 80, 255), max(2, half // 100))
        cv2.putText(ctx, b["class"], (int(ex - x0), max(12, int(ey - y0) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, half / 250), (255, 80, 255), max(1, half // 200))
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


def compose(img, ann, cats, keys, cur, cls_prog, total, n_done, counts=None, cap=0, msg="", prompt=None, hist=None, extra=None):
    """Image panels plus Ubuntu-font text, as a PIL image."""
    from PIL import Image, ImageDraw
    panels = Image.fromarray(cv2.cvtColor(render(img, ann, extra), cv2.COLOR_BGR2RGB))
    W = panels.width
    big, mid, small, tiny = ui_font(34, True), ui_font(22), ui_font(24), ui_font(20)
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    cls_lines = wrap(d, [f"{k}={n}" for n, k in keys.items()], small, W - 200)
    acc_lines = wrap(d, [f"{n} {counts.get(n, 0)}/{cap}" for n in keys], tiny, W - 140) if cap else []
    if hist:
        hints = ["LEFT / RIGHT  older / newer", "TAB  filter", "SPACE  confirm", "X  reject", "F  flag", "R  re-annotate",
                 "N  new class", "V  image view", "H  back to queue", "Q  quit"]
    else:
        hints = ["SPACE  confirm", "X  reject", "F  flag", "R  re-annotate", "S  skip", "U  undo", "TAB  next class",
                 "N  new class", "V  image view", "H  history", "Q  quit"]
    hint_lines = wrap(d, hints, small, W - 32)
    top_h = 96
    bot_h = 10 + 34 * len(hint_lines) + 34 * len(cls_lines) + (8 + 28 * len(acc_lines) if cap else 0) + 14
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
        txt = decision_text(cur)
        d.text((W - 16 - d.textlength(txt, font=mid), 14), txt, font=mid, fill=(255, 190, 80))
    if hist:
        d.text((16, 58), f"HISTORY {hist[0]} / {hist[1]}  [{hist[2]}]  (1 = oldest)    {ann['file']}", font=tiny, fill=(215, 190, 255))
    else:
        d.text((16, 58), f"{name} {cls_prog[0]} / {cls_prog[1]}    all {n_done} / {total}    {ann['file']}", font=tiny, fill=(150, 150, 150))
    y = top_h + panels.height + 10
    for ln in hint_lines:
        d.text((16, y), ln, font=small, fill=(90, 255, 120))
        y += 34
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


def corrected_coco(data, cats, dec, added=()):
    """COCO dict with only the verified objects: confirmed / re-classed originals, re-annotated boxes, added boxes.
    Rejected, flagged and undecided detections are left out. Re-annotated and added boxes have no segmentation."""
    cid = {n: i for i, n in cats.items()}
    anns = {str(a["id"]): a for a in data["annotations"]}
    out = []
    nid = max((a["id"] for a in data["annotations"]), default=0) + 1

    def cat(name):
        return cid.setdefault(name, max(cid.values(), default=0) + 1)

    def new(image_id, bbox, name, source, parent=None, score=None):
        nonlocal nid
        x, y, w, h = bbox
        a = {"id": nid, "image_id": image_id, "category_id": cat(name), "bbox": [x, y, w, h], "area": w * h,
             "iscrowd": 0, "source": source}
        if parent is not None:
            a["parent_id"] = parent
        if score is not None:
            a["score"] = score
        nid += 1
        out.append(a)

    for aid in sorted(dec, key=lambda k: (len(k), k)):
        d, a = dec[aid], anns.get(aid)
        if a is None:
            continue
        if d["action"] in ("confirm", "change"):
            c = {k: v for k, v in a.items() if k != "file"}
            c["category_id"] = cat(d["class"])
            out.append(c)
        elif d["action"] == "redraw":
            for b in d["boxes"]:
                new(a["image_id"], b["bbox"], b["class"], "redraw", a["id"], a.get("score"))
    for ad in added:
        new(ad["image_id"], ad["bbox"], ad["class"], "added")
    res = {k: v for k, v in data.items() if k not in ("images", "annotations", "categories")}
    res["images"] = data["images"]
    res["categories"] = [{"id": i, "name": n, "supercategory": ""} for n, i in sorted(cid.items(), key=lambda kv: kv[1])]
    res["annotations"] = out
    return res


def export(args, data, cats, dec, added=()):
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
    jobs = {}   # image id -> [(target dir, class, bbox, pad fraction, file suffix)]
    flagged = []
    for aid, d in dec.items():
        a = anns.get(aid)
        if a is None:
            continue
        if d["action"] in ("confirm", "change"):
            jobs.setdefault(a["image_id"], []).append((out, d["class"], a["bbox"], 0, aid))
        elif d["action"] == "redraw":
            for k, b in enumerate(d["boxes"]):
                jobs.setdefault(a["image_id"], []).append((out, b["class"], b["bbox"], 0, f"{aid}_{k}"))
        elif d["action"] == "flag":
            flagged.append((aid, a))
            # flagged objects are cropped with extra context so a bad box is easy to see
            jobs.setdefault(a["image_id"], []).append((fdir, cats[a["category_id"]], a["bbox"], 0.5, aid))
    for ad in added:
        jobs.setdefault(ad["image_id"], []).append((out, ad["class"], ad["bbox"], 0, ad["id"]))
    n = 0
    for iid, items in jobs.items():
        im = cv2.imread(str(Path(args.dataset) / "images" / imgs[iid]["file_name"]))
        if im is None:
            print(f"warning: cannot read {imgs[iid]['file_name']}, {len(items)} crops skipped")
            continue
        stem = Path(imgs[iid]["file_name"]).stem
        for target, cls, bbox, frac, suffix in items:
            crop, _ = crop_box(im, bbox, frac, args.pad)
            cdir = target / cls
            cdir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cdir / f"{stem}_{suffix}.png"), crop)
            n += target is out
    if not flagged:
        (fdir / "flagged.json").unlink(missing_ok=True)
    else:
        fdir.mkdir(parents=True, exist_ok=True)
        (fdir / "flagged.json").write_text(json.dumps([
            {"annotation_id": a["id"], "image": imgs[a["image_id"]]["file_name"], "bbox": a["bbox"],
             "predicted": cats[a["category_id"]], "score": a.get("score")} for _, a in flagged], indent=1))
    coco = corrected_coco(data, cats, dec, added)
    save_json(out / "corrected_instances.json", coco)
    print(f"exported {n} crops to {out}/" + (f", {len(flagged)} flagged to {fdir}/" if flagged else "")
          + f", {len(coco['annotations'])} verified annotations to {out}/corrected_instances.json")


def main():
    args = parse_args()
    ds = Path(args.dataset)
    data = json.loads((ds / "annotations" / "instances.json").read_text())
    cats, label_only, keys = load_classes(args.classes, data["categories"])
    imgs = {i["id"]: i for i in data["images"]}
    out = Path(args.out)
    dec = load_state(out)
    added_file = out / "added.json"
    added = json.loads(added_file.read_text()) if added_file.exists() else []   # objects drawn in the viewer

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

    # extra classes added during earlier sessions: [{"name":..., "key":...}]
    extra_file = out / "classes.json"
    extra = json.loads(extra_file.read_text()) if extra_file.exists() else []
    extra = [e for e in extra if e["name"] not in keys]  # now defined in classes.yaml
    for e in extra:
        if e["key"] in keys.values() or e["key"] in RESERVED:  # key now taken (yaml class / command key): move this one
            e["key"] = next(k for k in KEY_POOL if k not in keys.values() and k not in {x["key"] for x in extra})
            extra_file.write_text(json.dumps(extra))
            print(f"class {e['name']!r} moved to key {e['key']!r} (its old key is used in classes.yaml or by a command)")
        cats[NEW_ID + len(cats)] = e["name"]
        keys[e["name"]] = e["key"]
    for n in list(keys):  # classes without a key in the yaml get the next free one
        if keys[n] is None:
            keys[n] = next(k for k in KEY_POOL if k not in keys.values())

    if args.export_only:
        export(args, data, cats, dec, added)
        return

    recheck = [PSEUDO.get({"rejected": "reject", "flagged": "flag", "redrawn": "redraw"}.get(n, ""), n) for n in args.recheck]
    bad = [n for n in recheck if n not in keys and n not in PSEUDO_ORDER]
    if bad:
        sys.exit(f"--recheck: unknown class {bad}; known: {list(keys)} + rejected, flagged, redrawn")

    def is_open(a):  # still needs a decision (in --only-flagged mode: still carries the flag)
        d = dec.get(str(a["id"]))
        return d is not None and d["action"] == "flag" if args.only_flagged else d is None

    pending = [i for i, a in enumerate(anns) if is_open(a)]
    ann_idx = {str(a["id"]): i for i, a in enumerate(anns)}
    if not pending and not any(k in ann_idx for k in dec) and not args.view:
        print("nothing left to review")
        export(args, data, cats, dec, added)
        return

    import tkinter as tk
    from PIL import Image, ImageTk
    from editor import BoxEditor
    from viewer import ImageViewer

    skipped = set()
    history = []  # annotation ids in order of decision, for undo
    cache = {"file": None, "img": None}
    # st["hist"]: annotation id shown in history mode; st["filter"]/["flabel"]: history class filter (None = all)
    st = {"pos": 0, "cur": None, "msg": "", "prompt": None, "hist": None, "filter": None, "flabel": "all",
          "mode": "review", "back": "review", "edit": None}
    cap = args.max_per_class
    by_img, coco_names = {}, {c["id"]: c["name"] for c in data["categories"]}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    def counts():
        c = {}
        for d in dec.values():
            for n in accepted_classes(d):
                c[n] = c.get(n, 0) + 1
        for ad in added:
            c[ad["class"]] = c.get(ad["class"], 0) + 1
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
            if cache["img"] is None:
                raise SystemExit(f"cannot read image {ds / 'images' / f}")
            cache["file"] = f
        return cache["img"]

    root = tk.Tk()
    root.title(WIN)
    from tkinter import font as tkfont
    for f in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):  # bigger widget text for the editor / viewer
        tkfont.nametofont(f).configure(size=round(13 * args.ui_scale))
    tkfont.nametofont("TkFixedFont").configure(size=round(12 * args.ui_scale))
    import boxcanvas
    boxcanvas.SCALE = args.ui_scale
    global PANEL
    PANEL = max(300, min(PANEL, (root.winfo_screenheight() - 460), (root.winfo_screenwidth() - 60) // 2))
    root.configure(bg="black")
    label = tk.Label(root, bd=0, bg="black")
    label.pack()
    big_geometry = f"{root.winfo_screenwidth() - 80}x{root.winfo_screenheight() - 120}+40+40"

    def finish():
        save_state(out, dec)
        root.destroy()

    def apply_decision(aid, d):
        history[:] = [h for h in history if h[0] != aid]  # keep the undo stack consistent
        dec[aid] = d
        save_state(out, dec)

    def create_class(raw):
        """New class from a typed name -> (name, "") or (None, reason)."""
        name = "_".join(raw.split()).replace("/", "-").replace("\\", "-")
        if not name or name.startswith("."):
            return None, "invalid name"
        if name.lower() in (n.lower() for n in keys):
            return None, f"{name} already exists"
        free = next((k for k in KEY_POOL if k not in keys.values()), None)
        if free is None:
            return None, "no free shortcut keys left"
        cats[NEW_ID + len(cats)] = name
        keys[name] = free
        extra.append({"name": name, "key": free})
        save_json(extra_file, extra)
        return name, ""

    # ---- history (with class filter)
    def hist_ids():  # decided objects matching the filter, oldest first
        flt = st["filter"]
        return [k for k, d in dec.items() if k in ann_idx and (flt is None or decision_tags(d) & flt)]

    def filter_options():
        present = set()
        for k, d in dec.items():
            if k in ann_idx:
                present |= decision_tags(d)
        opts = [("all", None)]
        if len(recheck) > 1:
            opts.append(("+".join(recheck), set(recheck)))
        return opts + [(n, {n}) for n in [n for n in keys if n in present] + [p for p in PSEUDO_ORDER if p in present]]

    def reset_filter():
        st["filter"], st["flabel"] = None, "all"

    def cycle_filter(delta):
        opts = filter_options()
        labels = [o[0] for o in opts]
        i = labels.index(st["flabel"]) if st["flabel"] in labels else 0
        st["flabel"], st["filter"] = opts[(i + delta) % len(opts)]
        ids = hist_ids()
        st["hist"] = ids[0] if ids else None
        st["msg"] = f"filter: {st['flabel']}"
        show()

    def enter_history(aid=None, msg="", oldest=False):
        if not hist_ids() and st["filter"] is not None:
            reset_filter()
            msg = msg or "nothing decided for that filter: showing all"
        ids = hist_ids()
        if not ids:
            st["msg"] = "no decisions yet"
            return False
        st["hist"] = aid if aid in ids else (ids[0] if oldest else ids[-1])
        st["msg"] = msg
        return True

    def show_history():
        ids = hist_ids()
        if not ids and st["filter"] is not None:
            reset_filter()
            ids = hist_ids()
        if st["hist"] not in ids:
            st["hist"] = ids[-1] if ids else None
        if st["hist"] is None:
            show()
            return
        i = ann_idx[st["hist"]]
        st["cur"] = i
        a = anns[i]
        d = dec.get(st["hist"])
        photo = ImageTk.PhotoImage(compose(get_img(a["file"]), a, cats, keys, d, None, len(anns), len(dec),
                                           counts(), cap, st["msg"], st["prompt"],
                                           (ids.index(st["hist"]) + 1, len(ids), st["flabel"]),
                                           d["boxes"] if d["action"] == "redraw" else None))
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
        d = dec.get(str(a["id"]))
        photo = ImageTk.PhotoImage(compose(get_img(a["file"]), a, cats, keys, d, cls_progress(a), len(anns), len(dec), c, cap,
                                           st["msg"], st["prompt"], None, d["boxes"] if d and d["action"] == "redraw" else None))
        st["msg"] = ""
        label.configure(image=photo)
        label.image = photo

    def decide(d):
        aid = str(anns[st["cur"]]["id"])
        if st["hist"] is not None:  # editing an earlier decision: overwrite in place, stay on this object
            before = hist_ids()
            k = before.index(aid) if aid in before else 0
            apply_decision(aid, d)
            st["msg"] = ""
            ids = hist_ids()
            if aid not in ids:  # no longer matches the filter: move on to the next object that does
                if ids:
                    st["hist"] = ids[min(k, len(ids) - 1)]
                else:
                    reset_filter()
            show()
            return
        history.append((aid, dec.get(aid)))
        dec[aid] = d
        save_state(out, dec)
        st["pos"] += 1
        show()

    def class_decision(a, name):
        """Decision for 'give object a this class', or (None, reason) when the class is full."""
        own = dec.get(str(a["id"]), {})
        own = own.get("class") if own.get("action") in ("confirm", "change") else None
        if name != own and full(name, counts()):
            return None, f"{name} is full"
        return {"action": "confirm" if name == cats[a["category_id"]] else "change", "class": name}, ""

    def accept(name):
        d, msg = class_decision(anns[st["cur"]], name)
        if d is None:
            st["msg"] = msg
            show()
            return
        decide(d)

    def add_class(raw):
        name, st["msg"] = create_class(raw)
        if name:
            accept(name)
        else:
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

    # ---- editor and viewer modes
    def switch(mode):
        frames[st["mode"]].pack_forget()
        st["mode"] = mode
        if mode == "review":
            label.pack()
            root.geometry("")
        else:
            frames[mode].pack(fill="both", expand=True)
            root.geometry(big_geometry)

    def next_added_id():
        return "n" + str(max((int(x["id"][1:]) for x in added), default=0) + 1)

    def on_edit_done(result):
        a, back = st["edit"], st["back"]
        aid = str(a["id"])
        switch(back)
        if result is None:
            if back == "review":
                show()
            else:
                viewer.reload()
            return
        d = {"action": "redraw", "boxes": result} if result else {"action": "reject"}
        if back == "review":
            decide(d)
        else:
            apply_decision(aid, d)
            viewer.reload()

    def open_editor(a, back):
        d = dec.get(str(a["id"]))
        pred = cats[a["category_id"]]
        if d and d["action"] == "redraw":
            boxes = d["boxes"]
        else:
            boxes = [{"bbox": a["bbox"], "class": d["class"] if d and d["action"] in ("confirm", "change") else pred}]
        neigh = [b["bbox"] for b in by_img.get(a["image_id"], []) if b is not a and b["category_id"] in cats]
        st["edit"], st["back"] = a, back
        switch("editor")
        editor.start(get_img(a["file"]), a, boxes, boxes[0]["class"], neigh,
                     f"{a['file']}  ·  predicted {pred}  ·  id {a['id']}")

    removed_cache = {}

    def removed_by_img():
        if not removed_cache:
            f = ds / "annotations" / "removed_pedestrians.json"
            for r in (json.loads(f.read_text()) if f.exists() else []):
                removed_cache.setdefault(r["image_id"], []).append(r)
            removed_cache[None] = []
        return removed_cache

    def viewer_items(iid):
        items = []
        for a in by_img.get(iid, []):
            aid = str(a["id"])
            if aid not in ann_idx:  # not reviewed: other category or filtered by --min-score / --min-size
                items.append(dict(id=aid, bbox=a["bbox"], cls=coco_names.get(a["category_id"], "?"), status="ignored",
                                  score=a.get("score"), kind="ignored"))
                continue
            d, pred = dec.get(aid), cats[a["category_id"]]
            if d and d["action"] == "redraw":
                for k, b in enumerate(d["boxes"]):
                    items.append(dict(id=f"{aid}#{k}", bbox=b["bbox"], cls=b["class"], status="redrawn", kind="sub", parent=aid))
                continue
            status = {None: "undecided", "confirm": "confirmed", "change": "changed", "reject": "rejected",
                      "flag": "flagged"}[d["action"] if d else None]
            items.append(dict(id=aid, bbox=a["bbox"], cls=d["class"] if d and "class" in d else pred, status=status,
                              score=a.get("score"), kind="det"))
        for r in removed_by_img().get(iid, []):
            items.append(dict(id=f"r{r['id']}", bbox=r["bbox"], cls="Pedestrian", status="removed", score=r.get("score"),
                              kind="removed"))
        for ad in added:
            if ad["image_id"] == iid:
                items.append(dict(id=ad["id"], bbox=ad["bbox"], cls=ad["class"], status="added", kind="added"))
        return items

    def viewer_apply(it, op, cls):
        kind = it["kind"]
        if kind == "added":
            ad = next(x for x in added if x["id"] == it["id"])
            if op == "class":
                ad["class"] = cls
            elif op == "reject":
                added.remove(ad)
            else:
                return "added boxes can only change class or be deleted (X)"
            save_json(added_file, added)
            return "added box deleted" if op == "reject" else ""
        if kind == "sub":
            return "re-annotated box: press R to edit it again"
        if kind != "det":
            return "not part of the review (other category, or filtered by --min-score / --min-size)"
        a = anns[ann_idx[it["id"]]]
        if op == "class":
            d, msg = class_decision(a, cls)
            if d is None:
                return msg
        else:
            d = {"confirm": {"action": "confirm", "class": cats[a["category_id"]]}, "reject": {"action": "reject"},
                 "flag": {"action": "flag"}}[op]
        apply_decision(it["id"], d)
        return ""

    def viewer_add(image_id, b, cls):
        if full(cls, counts()):
            return f"{cls} is full"
        added.append({"id": next_added_id(), "image_id": image_id, "class": cls,
                      "bbox": [round(b[0]), round(b[1]), round(b[2] - b[0]), round(b[3] - b[1])]})
        save_json(added_file, added)
        return f"added {cls}"

    def viewer_edit(it):
        if it["kind"] not in ("det", "sub"):
            return "only reviewed detections can be re-annotated"
        open_editor(anns[ann_idx[it.get("parent", it["id"])]], "viewer")
        return ""

    def leave_viewer():
        switch("review")
        show()

    def open_viewer(image_id, select=None):
        switch("viewer")
        viewer.open(image_id, select)

    host = SimpleNamespace(images=sorted(data["images"], key=lambda i: i["id"]), keys=keys, items=viewer_items,
                           get_img=get_img, apply=viewer_apply, add=viewer_add, edit=viewer_edit,
                           new_class=create_class, leave=leave_viewer, quit=finish)
    editor = BoxEditor(root, keys, create_class, on_edit_done)
    viewer = ImageViewer(root, host)
    frames = {"review": label, "editor": editor, "viewer": viewer}

    def on_key(e):
        if isinstance(e.widget, tk.Entry):  # typing in the viewer's "go to" box
            return
        if st["mode"] != "review":
            frames[st["mode"]].on_key(e)
            return
        ks = e.keysym
        ch = e.char if e.char and e.char.isprintable() and e.char != " " else ""
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
            if ks in ("Left", "Right", "Home", "End", "Prior", "Next"):
                ids = hist_ids()
                k = ids.index(st["hist"])
                k = {"Left": k - 1, "Right": k + 1, "Prior": k - 10, "Next": k + 10, "Home": 0, "End": len(ids) - 1}[ks]
                if 0 <= k < len(ids):
                    st["hist"] = ids[k]
                else:
                    st["msg"] = "oldest decision" if k < 0 else "newest decision"
                    st["hist"] = ids[0] if k < 0 else ids[-1]
                show()
                return
            if ks in ("Tab", "ISO_Left_Tab", "Shift_Tab"):
                cycle_filter(1 if ks == "Tab" else -1)
                return
            if ks in ("s", "u", "BackSpace"):
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
        elif ks == "r":
            open_editor(a, "review")
        elif ks == "v":
            open_viewer(a["image_id"], str(a["id"]))
        elif by_key.get(ch) or by_key.get(ks):
            accept(by_key.get(ch) or by_key.get(ks))
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
    if recheck:
        recheck_set = set(recheck)
        st["flabel"], st["filter"] = "+".join(recheck), recheck_set
    if args.view:
        open_viewer(host.images[0]["id"])
    else:
        if args.history or recheck or not pending:
            enter_history(msg="" if args.history or recheck else "queue is empty: history mode, Q to quit", oldest=bool(recheck))
        show()
    try:
        root.mainloop()
    except tk.TclError:  # everything already capped: window closed before the loop started
        pass

    left = sum(1 for a in anns if str(a["id"]) not in dec)
    print(f"decided {len(dec)}, remaining {left}")
    export(args, data, cats, dec, added)


if __name__ == "__main__":
    sys.exit(main())
