"""Build an Ultralytics OBB dataset from a folder of images using a trained model's (tiled) predictions.

Each image is resized to --max-side, tiled with overlap if larger than --imgsz (see infer_tiled.detect),
and the merged detections are written as normalized OBB labels (`cls x1 y1 x2 y2 x3 y3 x4 y4`, classes as in
common.CLASSES). Images are copied at native resolution (labels are normalized, so scale-free).
Train/val split is by contiguous blocks of sorted file names (--val-frac), since neighbouring drone frames are
near-duplicates. These are PSEUDO-labels (model output, not human-verified): raise --conf to trade recall for precision.

Usage: python pseudo_label.py --weights best.pt --source /tmp/b1/images --out /tmp/b1_pseudo_obb --conf 0.5 --device 0
Outputs: images/{train,val}, labels/{train,val}, data.yaml, stats.json
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from common import CLASSES
from infer_tiled import detect


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.5, help="pseudo-label confidence threshold")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-side", type=int, default=1920)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.3)
    ap.add_argument("--edge", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    src, out = Path(a.source), Path(a.out)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    n_val = max(1, round(len(files) * a.val_frac))
    model = YOLO(a.weights)
    stats = {"images": {"train": 0, "val": 0}, "instances": {c: 0 for c in CLASSES}, "conf": a.conf, "weights": a.weights}
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    for k, p in enumerate(files, 1):
        sp = "val" if k > len(files) - n_val else "train"
        im = cv2.imread(str(p))
        NH, NW = im.shape[:2]
        s = min(1.0, a.max_side / max(NW, NH))
        work = cv2.resize(im, (round(NW * s), round(NH * s)), interpolation=cv2.INTER_AREA) if s < 1 else im
        H, W = work.shape[:2]
        boxes, scores, cls, _ = detect(model, work, a.imgsz, a.overlap, a.conf, a.nms_iou, a.edge, a.device, a.batch)
        lines = []
        for (cx, cy, w, h, ang), c in zip(boxes.tolist(), cls.tolist()):
            poly = np.clip(cv2.boxPoints(((cx, cy), (w, h), np.degrees(ang))) / [W, H], 0, 1)
            lines.append(f"{int(c)} " + " ".join(f"{v:.6f}" for v in poly.reshape(-1)))
            stats["instances"][model.names[int(c)]] += 1
        shutil.copy2(p, out / "images" / sp / p.name)
        (out / "labels" / sp / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["images"][sp] += 1
        print(f"[{k}/{len(files)}] {p.name} -> {sp}, {len(lines)} boxes", flush=True)
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (out / "data.yaml").write_text(f"path: .\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
