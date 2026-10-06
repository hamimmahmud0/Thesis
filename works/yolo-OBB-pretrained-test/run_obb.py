"""Smoke test: pretrained Ultralytics YOLO11 OBB (DOTAv1) inference on sample aerial images.

Usage: python run_obb.py [--weights yolo11n-obb.pt] [--source IMG_OR_DIR ...] [--out out]
Writes annotated images and a results.json (class, conf, rotated box xywhr, polygon) to --out.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n-obb.pt")
    ap.add_argument("--source", nargs="*", default=[str(ASSETS / "boats.jpg")])
    ap.add_argument("--out", default="out")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == 0 else "")

    model = YOLO(args.weights)
    print("task:", model.task, "| classes:", len(model.names))

    results_json = {}
    for src in args.source:
        t0 = time.time()
        results = model.predict(src, imgsz=args.imgsz, conf=args.conf, device=device,
                                save=True, project=str(out), name="pred", exist_ok=True, verbose=False)
        dt = time.time() - t0
        for r in results:
            obb = r.obb
            dets = []
            if obb is not None and len(obb):
                for c, p, xywhr, poly in zip(obb.cls.tolist(), obb.conf.tolist(),
                                             obb.xywhr.tolist(), obb.xyxyxyxy.tolist()):
                    dets.append({"cls": model.names[int(c)], "conf": round(p, 4),
                                 "xywhr": [round(v, 2) for v in xywhr],
                                 "poly": [[round(v, 1) for v in pt] for pt in poly]})
            results_json[Path(r.path).name] = {"n": len(dets), "sec": round(dt, 3), "dets": dets}
            counts = {}
            for d in dets:
                counts[d["cls"]] = counts.get(d["cls"], 0) + 1
            print(Path(r.path).name, "->", len(dets), "dets", counts, f"({dt:.2f}s)")

    (out / "results.json").write_text(json.dumps(results_json, indent=1))


if __name__ == "__main__":
    main()
