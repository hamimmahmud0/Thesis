"""Run trained OBB weights on a folder of images, tiling any image larger than --imgsz.

Pipeline per image: resize so the long side is --max-side (default 1920, the size training images were
prepared at; no upscaling) -> if the result is larger than --imgsz, cut overlapping --imgsz tiles, run
each at imgsz, drop boxes cut by interior tile borders, shift back, merge with rotated NMS -> scale
boxes to native resolution. Smaller images are predicted in a single pass.

Usage (kaggle-vm):
  python infer_tiled.py --weights best.pt --source /tmp/b1/images --out /kaggle/working/b1_pred --device 1
Outputs: <out>/annotated/*.jpg (boxes drawn on the --max-side image), detections.json
(per image: native size, tiles used, list of {cls, conf, xywhr (native px), poly}), summary.json (class counts).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.nms import TorchNMS

from common import CLASSES

rng = np.random.RandomState(1)
COLORS = {c: tuple(int(v) for v in rng.randint(60, 255, 3)) for c in CLASSES}


def starts(length, tile, stride):
    if length <= tile:
        return [0]
    return list(range(0, length - tile, stride)) + [length - tile]


def detect(model, img, imgsz, overlap, conf, nms_iou, edge, device, batch):
    """Return (boxes xywhr, scores, cls) tensors in `img` pixel coordinates, tiling if needed."""
    H, W = img.shape[:2]
    stride = int(imgsz * (1 - overlap))
    xs, ys = starts(W, imgsz, stride), starts(H, imgsz, stride)
    tiles = [(x, y) for y in ys for x in xs]
    boxes, scores, cls = [], [], []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        crops = [img[y:y + imgsz, x:x + imgsz] for x, y in chunk]
        res = model.predict(crops, imgsz=imgsz, conf=conf, device=device, half=True, verbose=False)
        for (x, y), r in zip(chunk, res):
            if r.obb is None or not len(r.obb):
                continue
            b = r.obb.xywhr.cpu().clone()
            p = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 8)
            th, tw = r.orig_shape
            px, py = p[:, 0::2], p[:, 1::2]
            cut = np.zeros(len(p), bool)
            if x > 0: cut |= px.min(1) < edge
            if x + tw < W: cut |= px.max(1) > tw - edge
            if y > 0: cut |= py.min(1) < edge
            if y + th < H: cut |= py.max(1) > th - edge
            keep = torch.from_numpy(~cut)
            b[:, 0] += x
            b[:, 1] += y
            boxes.append(b[keep]); scores.append(r.obb.conf.cpu()[keep]); cls.append(r.obb.cls.cpu()[keep])
    if not boxes or not sum(len(b) for b in boxes):
        return torch.zeros(0, 5), torch.zeros(0), torch.zeros(0), len(tiles)
    boxes, scores, cls = torch.cat(boxes), torch.cat(scores), torch.cat(cls)
    if len(tiles) > 1:
        keep = TorchNMS.fast_nms(boxes, scores, nms_iou, iou_func=batch_probiou)
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    return boxes, scores, cls, len(tiles)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image folder (jpg/jpeg/png, case-insensitive)")
    ap.add_argument("--out", default="pred_tiled")
    ap.add_argument("--imgsz", type=int, default=1280, help="tile size / model input size")
    ap.add_argument("--max-side", type=int, default=1920, help="resize long side to this before tiling (0 = native)")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.3)
    ap.add_argument("--edge", type=float, default=3.0, help="drop boxes within N px of an interior tile border")
    ap.add_argument("--batch", type=int, default=4, help="tiles per forward pass")
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="only first N images (0 = all)")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(a.source).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if a.limit:
        files = files[:a.limit]
    model = YOLO(a.weights)
    names = model.names
    dets_all, counts, t0 = {}, {}, time.time()
    for n, p in enumerate(files, 1):
        native = cv2.imread(str(p))
        NH, NW = native.shape[:2]
        s = min(1.0, a.max_side / max(NW, NH)) if a.max_side else 1.0
        img = cv2.resize(native, (round(NW * s), round(NH * s)), interpolation=cv2.INTER_AREA) if s < 1 else native
        boxes, scores, cls, ntiles = detect(model, img, a.imgsz, a.overlap, a.conf, a.nms_iou, a.edge, a.device, a.batch)
        dets = []
        for (cx, cy, w, h, ang), sc, c in zip(boxes.tolist(), scores.tolist(), cls.tolist()):
            name = names[int(c)]
            poly = cv2.boxPoints(((cx, cy), (w, h), np.degrees(ang)))
            cv2.polylines(img, [poly.astype(np.int32)], True, COLORS[name], 2)
            cv2.putText(img, name, tuple(poly[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[name], 1)
            counts[name] = counts.get(name, 0) + 1
            f = 1 / s
            dets.append({"cls": name, "conf": round(sc, 3), "xywhr": [round(v * f, 2) for v in (cx, cy, w, h)] + [round(ang, 4)],
                         "poly": [[round(x * f, 1), round(y * f, 1)] for x, y in poly.tolist()]})
        cv2.imwrite(str(out / "annotated" / (p.stem + ".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        dets_all[p.name] = {"native_size": [NW, NH], "work_size": [img.shape[1], img.shape[0]], "tiles": ntiles, "dets": dets}
        print(f"[{n}/{len(files)}] {p.name} {NW}x{NH} -> {img.shape[1]}x{img.shape[0]}, {ntiles} tile(s), {len(dets)} dets "
              f"({(time.time() - t0) / n:.1f}s/img)", flush=True)
    (out / "detections.json").write_text(json.dumps(dets_all))
    (out / "summary.json").write_text(json.dumps({"images": len(files), "class_counts": counts, "weights": a.weights,
                                                  "imgsz": a.imgsz, "max_side": a.max_side}, indent=1))
    print("class counts:", counts)


if __name__ == "__main__":
    main()
