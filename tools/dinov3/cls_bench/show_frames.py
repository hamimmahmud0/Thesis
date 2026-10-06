#!/usr/bin/env python3
"""Draw ALL detections (final class) on a few full frames of the re-annotated dataset.
Box colour = final class; thick box = class changed (label 'new (was old)'); '*' = human label."""
import argparse, json, subprocess
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def dedup(anns, iou_t, ioa_t):
    """Greedy overlap removal on the segmentation masks: of two overlapping detections the SMALLER (mask area) is dropped."""
    import pycocotools.mask as mu
    if len(anns) < 2:
        return anns
    ar0 = [float(mu.area(x["segmentation"])) for x in anns]
    idx = sorted(range(len(anns)), key=lambda i: (-ar0[i], anns[i]["label_source"] != "human", -anns[i]["model_conf"]))   # biggest first
    order = [anns[i] for i in idx]
    rl = [x["segmentation"] for x in order]
    area = np.array([mu.area(r) for r in rl], float)
    M = mu.iou(rl, rl, [0] * len(rl))
    keep = []
    for i in range(len(order)):
        ok = True
        for j in keep:
            iou = M[i, j]
            inter = iou * (area[i] + area[j]) / (1 + iou)
            if iou >= iou_t or inter / max(min(area[i], area[j]), 1) >= ioa_t:
                ok = False
                break
        if ok:
            keep.append(i)
    return [order[i] for i in keep]


ap = argparse.ArgumentParser()
ap.add_argument("--bucket", default="hf://buckets/z81980440/SAM_COCO_b1_reannotated/b1")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--frames", nargs="*", default=[], help="file names to draw (default: 4 frames by detection density)")
ap.add_argument("--width", type=int, default=3200)
ap.add_argument("--exclude", nargs="*", default=[], help="final classes not to draw, e.g. not_an_object")
ap.add_argument("--suffix", default="detections")
ap.add_argument("--dedup", action="store_true", help="drop overlapping detections: the smaller one of each overlapping pair is removed")
ap.add_argument("--iou", type=float, default=0.5, help="--dedup: mask IoU at or above this = overlap")
ap.add_argument("--ioa", type=float, default=0.8, help="--dedup: share of the smaller mask inside the other at or above this = overlap")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
subprocess.run(["hf", "buckets", "cp", f"{a.bucket}/annotations/instances.json", str(out / "instances.json")], check=True, capture_output=True)
d = json.load(open(out / "instances.json"))
cat = {c["id"]: c["name"] for c in d["categories"]}
old = {1: "Pedestrian", 2: "Vehicle", 3: "Car", 4: "Bus", 5: "Truck", 6: "Motorcycle", 7: "Bicycle", 8: "Rickshaw"}
cm = plt.get_cmap("tab20"); names = sorted(cat.values())
col = {n: tuple(int(255 * c) for c in cm(i % 20)[:3][::-1]) for i, n in enumerate(names)}
by = {}
for x in d["annotations"]:
    by.setdefault(x["image_id"], []).append(x)
imgs = sorted(d["images"], key=lambda i: len(by.get(i["id"], [])))
picks = [imgs[int(q * (len(imgs) - 1))] for q in [0.15, 0.4, 0.65, 0.9][:a.n]]
if a.frames:
    picks = [i for i in d["images"] if i["file_name"] in a.frames]
for im in picks:
    fn = im["file_name"]; p = out / fn
    subprocess.run(["hf", "buckets", "cp", f"{a.bucket}/images/{fn}", str(p)], check=True, capture_output=True)
    img = cv2.imread(str(p)); sc = a.width / img.shape[1]; img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    total = len(by[im["id"]]); anns = [x for x in by[im["id"]] if cat[x["category_id"]] not in a.exclude]; chg = 0
    n_after_exclude = len(anns)
    if a.dedup:
        anns = dedup(anns, a.iou, a.ioa)
    for x in sorted(anns, key=lambda x: -x["bbox"][2] * x["bbox"][3]):
        bx, by_, bw, bh = [v * sc for v in x["bbox"]]; new = cat[x["category_id"]]; was = old.get(x["orig_category_id"], "?")
        ch = new != was; chg += ch; c = col[new]
        cv2.rectangle(img, (int(bx), int(by_)), (int(bx + bw), int(by_ + bh)), c, 4 if ch else 2)
        if bw > 26:
            t = new + ("*" if x["label_source"] == "human" else "") + (f" (was {was})" if ch else "")
            y = max(14, int(by_) - 4)
            cv2.putText(img, t, (int(bx), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, t, (int(bx), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    x0 = 10
    for n in names:   # legend
        if any(cat[x["category_id"]] == n for x in anns):
            cv2.rectangle(img, (x0 - 4, 6), (x0 + 12 + 11 * len(n), 34), (0, 0, 0), -1)
            cv2.putText(img, n, (x0, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col[n], 2, cv2.LINE_AA); x0 += 24 + 11 * len(n)
    cv2.imwrite(str(out / f"{Path(fn).stem}_{a.suffix}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(fn, f"{len(anns)} drawn of {total} detections ({n_after_exclude - len(anns)} overlaps removed),", chg, "changed", flush=True)
