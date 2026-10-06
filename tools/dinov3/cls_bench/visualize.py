#!/usr/bin/env python3
"""Visualise the class changes of the re-annotated dataset (original SAM class -> final class).

viz/transition_matrix.png      original class x final class (counts and row %)
viz/class_counts.png           class counts before / after
viz/confidence.png             model confidence: kept vs changed labels
viz/changes/<A>__to__<B>.jpg   contact sheet of sampled detections that changed from A to B (conf shown)
viz/overlays/<image>.jpg       full frames with only the changed detections drawn, labelled "old -> new"
viz/index.html                 gallery linking all of the above
"""
import argparse, collections, html, json, random, subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def colors(names):
    cm = plt.get_cmap("tab20")
    return {n: tuple(int(255 * c) for c in cm(i % 20)[:3][::-1]) for i, n in enumerate(names)}      # BGR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--images", required=True, help="local image dir; missing frames are fetched from --src")
    ap.add_argument("--src", default="hf://buckets/hamimmahmud0/SAM_COCO_b1/b1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-sheet", type=int, default=30)
    ap.add_argument("--max-sheets", type=int, default=24)
    ap.add_argument("--overlays", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out); [(out / d).mkdir(parents=True, exist_ok=True) for d in ("changes", "overlays")]
    coco = json.load(open(a.instances))
    new = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {i["id"]: i for i in coco["images"]}
    anns = coco["annotations"]
    old_name = lambda x: "Vehicle" if x["orig_category_id"] == 2 else new.get(x["orig_category_id"], str(x["orig_category_id"]))
    for x in anns:
        x["_old"], x["_new"] = old_name(x), new[x["category_id"]]
    olds = ["Pedestrian", "Vehicle", "Car", "Bus", "Truck", "Motorcycle", "Bicycle", "Rickshaw"]
    news = [c["name"] for c in coco["categories"]]
    trans = collections.Counter((x["_old"], x["_new"]) for x in anns)
    changed = [x for x in anns if x["_old"] != x["_new"]]
    print(f"{len(changed)} of {len(anns)} detections changed class", flush=True)

    # ---- transition matrix
    M = np.array([[trans[(o, n)] for n in news] for o in olds], float)
    R = M / np.maximum(M.sum(1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(1, 2, figsize=(22, 7))
    for k, (D, title, fmt) in enumerate(((M, "detections (count)", "{:.0f}"), (R, "% of each original class", "{:.0f}"))):
        im = ax[k].imshow(np.log10(D + 1) if k == 0 else D, cmap="viridis")
        ax[k].set_xticks(range(len(news)), news, rotation=60, ha="right"); ax[k].set_yticks(range(len(olds)), olds)
        for (i, j), v in np.ndenumerate(D):
            if v >= (1 if k == 0 else 1):
                ax[k].text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color="w" if (np.log10(v + 1) if k == 0 else v) < (2.2 if k == 0 else 50) else "k")
        ax[k].set_xlabel("final class (after re-annotation)"); ax[k].set_ylabel("original SAM class"); ax[k].set_title(title)
        for i, o in enumerate(olds):          # frame the diagonal (class unchanged)
            if o in news:
                j = news.index(o); ax[k].add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="red", lw=2))
    fig.suptitle("Class changes: original -> re-annotated (red box = unchanged)", fontsize=14); plt.tight_layout()
    plt.savefig(out / "transition_matrix.png", dpi=130); plt.close()

    # ---- class counts before / after
    allc = sorted(set(olds) | set(news), key=lambda n: -sum(1 for x in anns if n in (x["_new"],)))
    b = [sum(1 for x in anns if x["_old"] == n) for n in allc]; f = [sum(1 for x in anns if x["_new"] == n) for n in allc]
    fig, ax = plt.subplots(figsize=(13, 5)); w = .4; xs = np.arange(len(allc))
    ax.bar(xs - w / 2, b, w, label="original (SAM)"); ax.bar(xs + w / 2, f, w, label="re-annotated")
    ax.set_xticks(xs, allc, rotation=45, ha="right"); ax.set_yscale("log"); ax.set_ylabel("detections (log)"); ax.legend(); ax.set_title("Class counts before / after")
    plt.tight_layout(); plt.savefig(out / "class_counts.png", dpi=130); plt.close()

    # ---- confidence
    mod = [x for x in anns if x["label_source"] == "model"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist([x["model_conf"] for x in mod if x["_old"] == x["_new"]], bins=40, alpha=.6, label="class kept")
    ax.hist([x["model_conf"] for x in mod if x["_old"] != x["_new"]], bins=40, alpha=.6, label="class changed")
    ax.set_xlabel("model top-1 probability"); ax.set_ylabel("detections"); ax.legend(); ax.set_title("Model confidence (model-labelled detections)")
    plt.tight_layout(); plt.savefig(out / "confidence.png", dpi=130); plt.close()

    # ---- frames
    cache = {}

    def frame(iid):
        if iid not in cache:
            if len(cache) > 6:
                cache.pop(next(iter(cache)))
            p = Path(a.images) / imgs[iid]["file_name"]
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["hf", "buckets", "cp", f"{a.src}/images/{imgs[iid]['file_name']}", str(p)], check=True, capture_output=True)
            cache[iid] = cv2.imread(str(p))
        return cache[iid]

    # ---- contact sheets per transition
    rng = random.Random(0)
    tops = [t for t, _ in trans.most_common() if t[0] != t[1]][:a.max_sheets]
    T, COLS = 150, 10
    sheets = []
    for (o, n) in tops:
        pool = [x for x in changed if (x["_old"], x["_new"]) == (o, n) and x["label_source"] == "model"]
        pick = sorted(rng.sample(pool, min(a.per_sheet, len(pool))), key=lambda x: (x["image_id"], -x["model_conf"]))
        tiles = []
        for x in pick:
            im = frame(x["image_id"]); H, W = im.shape[:2]; bx, by, bw, bh = x["bbox"]
            m = max(bw, bh) * 0.6 + 12; cx, cy = bx + bw / 2, by + bh / 2
            x0, y0, x1, y1 = int(max(0, cx - m - bw / 2)), int(max(0, cy - m - bh / 2)), int(min(W, cx + m + bw / 2)), int(min(H, cy + m + bh / 2))
            c = im[y0:y1, x0:x1].copy()
            cv2.rectangle(c, (int(bx - x0), int(by - y0)), (int(bx + bw - x0), int(by + bh - y0)), (0, 255, 255), max(1, c.shape[1] // 120))
            s = T / max(c.shape[:2]); c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))
            t = np.zeros((T + 16, T, 3), np.uint8); t[:c.shape[0], :c.shape[1]] = c
            cv2.putText(t, f"{x['model_conf']:.2f}  #{x['id']}", (2, T + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        rows = [np.hstack(tiles[i:i + COLS] + [np.zeros_like(tiles[0])] * (COLS - len(tiles[i:i + COLS]))) for i in range(0, len(tiles), COLS)]
        sheet = np.vstack(rows); head = np.full((34, sheet.shape[1], 3), 30, np.uint8)
        cv2.putText(head, f"{o}  ->  {n}   ({trans[(o, n)]} detections, {len(tiles)} random samples; yellow = SAM box)", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        name = f"{o}__to__{n}.jpg"
        cv2.imwrite(str(out / "changes" / name), np.vstack([head, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88]); sheets.append((o, n, name))
        print("sheet", name, flush=True)

    # ---- overlays: images with most changes
    col = colors(sorted(set(olds) | set(news)))
    per = collections.Counter(x["image_id"] for x in changed)
    ov = []
    for iid, k in per.most_common(a.overlays):
        im = frame(iid).copy(); sc = 2400 / im.shape[1]
        im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        for x in changed:
            if x["image_id"] != iid:
                continue
            bx, by, bw, bh = [v * sc for v in x["bbox"]]; c = col[x["_new"]]
            cv2.rectangle(im, (int(bx), int(by)), (int(bx + bw), int(by + bh)), c, 2)
            if bw * 1 > 22:
                lab = f"{x['_old']}>{x['_new']}"
                cv2.putText(im, lab, (int(bx), max(10, int(by) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(im, lab, (int(bx), max(10, int(by) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        name = Path(imgs[iid]["file_name"]).stem + ".jpg"
        cv2.imwrite(str(out / "overlays" / name), im, [cv2.IMWRITE_JPEG_QUALITY, 85]); ov.append((name, k)); print("overlay", name, k, flush=True)

    # ---- index
    h = ["<!doctype html><meta charset=utf-8><title>Class changes</title><style>body{font:14px sans-serif;margin:20px;max-width:1500px}img{max-width:100%;border:1px solid #ccc;margin:6px 0}</style>",
         f"<h1>Re-annotation: class changes</h1><p>{len(changed)} of {len(anns)} detections ({100 * len(changed) / len(anns):.1f}%) changed class; "
         f"{sum(1 for x in anns if x['label_source'] == 'human')} detections carry a human label (never changed).</p>",
         "<h2>Overview</h2><img src=transition_matrix.png><br><img src=class_counts.png><img src=confidence.png><h2>Samples per transition</h2>"]
    h += [f"<h3>{html.escape(o)} &rarr; {html.escape(n)} ({trans[(o, n)]})</h3><img src='changes/{html.escape(nm)}'>" for o, n, nm in sheets]
    h += ["<h2>Frames with most changes</h2>"] + [f"<h3>{nm} ({k} changed)</h3><img src='overlays/{nm}'>" for nm, k in ov]
    (out / "index.html").write_text("\n".join(h))
    print("done", flush=True)


if __name__ == "__main__":
    main()
