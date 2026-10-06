"""Overlapping-tile (sliced) YOLO11 OBB vehicle detection on a video.

Each full-res frame is cut into overlapping --tile px windows, every tile is run at imgsz=--tile
(the scale DOTA models are trained at), boxes are shifted back to frame coordinates and merged
with rotated NMS. Boxes cut by an interior tile border are dropped (the neighbouring tile sees the
whole object thanks to the overlap).
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

VEHICLES = ("small vehicle", "large vehicle")
COLORS = {"small vehicle": (0, 255, 0), "large vehicle": (0, 128, 255)}


def starts(length, tile, stride):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile, stride))
    return s + [length - tile]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11l-obb.pt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="out_tiled")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.3)
    ap.add_argument("--edge", type=float, default=3.0, help="drop boxes within N px of an interior tile border")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="resize the frame by this factor before tiling (DOTA vehicles are ~20-30px; 4K vehicles are ~100px)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.weights)
    classes = [i for i, n in model.names.items() if n in VEHICLES]

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    FW, FH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W, H = round(FW * args.scale), round(FH * args.scale)  # working (downscaled) size used for tiling
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    stride = int(args.tile * (1 - args.overlap))
    xs, ys = starts(W, args.tile, stride), starts(H, args.tile, stride)
    print(f"video {FW}x{FH} -> working {W}x{H} (scale {args.scale}) @ {fps:.2f}fps, {total} frames | tile {args.tile}, stride {stride}, "
          f"{len(xs)}x{len(ys)}={len(xs) * len(ys)} tiles/frame | vehicle ids {classes}", flush=True)

    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FW, FH))
    totals, t0 = {}, time.time()
    with open(out / "detections.jsonl", "w") as f:
        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            work = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA) if args.scale != 1 else frame
            tiles, offs = [], []
            for y in ys:
                for x in xs:
                    tiles.append(work[y:y + args.tile, x:x + args.tile])
                    offs.append((x, y))
            res = model.predict(tiles, imgsz=args.tile, conf=args.conf, classes=classes,
                                device=device, half=device == 0, verbose=False)
            boxes, scores, cls = [], [], []
            for (x, y), r in zip(offs, res):
                if r.obb is None or not len(r.obb):
                    continue
                b = r.obb.xywhr.cpu().clone()
                p = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 8)
                th, tw = r.orig_shape
                # interior borders = tile edges that are not the frame edge
                lo_x, hi_x = (x > 0), (x + tw < W)
                lo_y, hi_y = (y > 0), (y + th < H)
                px, py = p[:, 0::2], p[:, 1::2]
                cut = np.zeros(len(p), bool)
                if lo_x: cut |= px.min(1) < args.edge
                if hi_x: cut |= px.max(1) > tw - args.edge
                if lo_y: cut |= py.min(1) < args.edge
                if hi_y: cut |= py.max(1) > th - args.edge
                keep = torch.from_numpy(~cut)
                b[:, 0] += x
                b[:, 1] += y
                boxes.append(b[keep])
                scores.append(r.obb.conf.cpu()[keep])
                cls.append(r.obb.cls.cpu()[keep])
            dets = []
            if boxes and sum(len(b) for b in boxes):
                boxes, scores, cls = torch.cat(boxes), torch.cat(scores), torch.cat(cls)
                keep = TorchNMS.fast_nms(boxes, scores, args.nms_iou, iou_func=batch_probiou)
                boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
                boxes[:, :4] /= args.scale  # back to full-res frame coordinates
                for (cx, cy, w, h, a), s, c in zip(boxes.tolist(), scores.tolist(), cls.tolist()):
                    name = model.names[int(c)]
                    poly = cv2.boxPoints(((cx, cy), (w, h), np.degrees(a)))
                    totals[name] = totals.get(name, 0) + 1
                    dets.append({"cls": name, "conf": round(s, 3), "xywhr": [round(v, 2) for v in (cx, cy, w, h, a)]})
                    cv2.polylines(frame, [poly.astype(np.int32)], True, COLORS[name], 2)
            writer.write(frame)
            f.write(json.dumps({"frame": i, "dets": dets}) + "\n")
            if i % 20 == 0:
                print(f"frame {i}/{total}: {len(dets)} dets | {(i + 1) / (time.time() - t0):.2f} fps", flush=True)
    writer.release()
    print("total detections:", totals)


if __name__ == "__main__":
    main()
