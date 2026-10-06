"""Run trained OBB weights on a video (optionally only the first N seconds) with overlapped tiling.

Per frame: resize so the long side is --max-side (default 1920, the training prep size), then if larger
than --imgsz cut overlapping --imgsz tiles, detect, drop boxes cut by interior tile borders, rotated NMS
(see infer_tiled.detect). The annotated video is written at --max-side resolution.

Usage: python infer_video_tiled.py --weights best.pt --source clip.mp4 --seconds 20 --out /kaggle/working/vid_pred
Outputs: <out>/<video stem>_annotated.mp4, <video stem>_detections.jsonl (per frame boxes at working res), <stem>_summary.json
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from infer_tiled import COLORS, detect


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, nargs="+", help="one or more videos")
    ap.add_argument("--out", default="vid_pred")
    ap.add_argument("--seconds", type=float, default=20, help="process only the first N seconds (0 = all)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-side", type=int, default=1920)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--nms-iou", type=float, default=0.3)
    ap.add_argument("--edge", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    for src in a.source:
        stem = Path(src).stem
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        FW, FH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if a.seconds:
            total = min(total, int(a.seconds * fps))
        s = min(1.0, a.max_side / max(FW, FH))
        W, H = round(FW * s), round(FH * s)
        print(f"{stem}: {FW}x{FH}@{fps:.2f} -> {W}x{H}, {total} frames", flush=True)
        writer = cv2.VideoWriter(str(out / f"{stem}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        counts, t0 = {}, time.time()
        with open(out / f"{stem}_detections.jsonl", "w") as f:
            for i in range(total):
                ok, frame = cap.read()
                if not ok:
                    break
                img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA) if s < 1 else frame
                boxes, scores, cls, nt = detect(model, img, a.imgsz, a.overlap, a.conf, a.nms_iou, a.edge, a.device, a.batch)
                dets = []
                for (cx, cy, w, h, ang), sc, c in zip(boxes.tolist(), scores.tolist(), cls.tolist()):
                    name = model.names[int(c)]
                    poly = cv2.boxPoints(((cx, cy), (w, h), np.degrees(ang)))
                    cv2.polylines(img, [poly.astype(np.int32)], True, COLORS[name], 2)
                    counts[name] = counts.get(name, 0) + 1
                    dets.append({"cls": name, "conf": round(sc, 3), "xywhr": [round(cx, 1), round(cy, 1), round(w, 1), round(h, 1), round(ang, 4)]})
                writer.write(img)
                f.write(json.dumps({"frame": i, "dets": dets}) + "\n")
                if i % 25 == 0:
                    print(f"  frame {i}/{total}: {len(dets)} dets, {nt} tiles, {(i + 1) / (time.time() - t0):.2f} fps", flush=True)
        writer.release()
        (out / f"{stem}_summary.json").write_text(json.dumps({"frames": i + 1, "fps": fps, "size": [W, H], "class_counts": counts}, indent=1))
        print(stem, "detections:", counts, flush=True)


if __name__ == "__main__":
    main()
