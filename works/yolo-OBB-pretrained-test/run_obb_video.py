"""Pretrained YOLO11 OBB (DOTAv1) vehicle detection on a video.

Keeps only DOTA vehicle classes ('small vehicle', 'large vehicle'). Writes an annotated
video and per-frame detections (detections.jsonl) to --out.
"""
import argparse
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

VEHICLES = ("small vehicle", "large vehicle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11l-obb.pt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="out_video")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.weights)
    classes = [i for i, n in model.names.items() if n in VEHICLES]
    print("device:", device, "| vehicle class ids:", classes)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"video {w}x{h} @ {fps:.2f}fps, {total} frames")

    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    totals = {}
    with open(out / "detections.jsonl", "w") as f:
        stream = model.predict(args.source, stream=True, imgsz=args.imgsz, conf=args.conf,
                               classes=classes, device=device, half=device == 0, verbose=False)
        for i, r in enumerate(stream):
            writer.write(r.plot(line_width=2, labels=False))
            dets = []
            if r.obb is not None and len(r.obb):
                for c, p, poly in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xyxyxyxy.tolist()):
                    name = model.names[int(c)]
                    totals[name] = totals.get(name, 0) + 1
                    dets.append({"cls": name, "conf": round(p, 3),
                                 "poly": [[round(v, 1) for v in pt] for pt in poly]})
            f.write(json.dumps({"frame": i, "dets": dets}) + "\n")
            if i % 100 == 0:
                print(f"frame {i}/{total}: {len(dets)} dets", flush=True)
    writer.release()
    print("total detections:", totals)


if __name__ == "__main__":
    main()
