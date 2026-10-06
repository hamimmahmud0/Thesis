"""Evaluate trained OBB weights on a split (full val or test): mAP50, mAP50-95, per-class AP.

Usage: python evaluate.py --weights /kaggle/working/runs/m1280/weights/best.pt --split test --imgsz 1280
Writes <out>/metrics_<split>.json.
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="/tmp/drashti/data_fullval.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--out", default=None, help="default: next to the weights' run dir")
    a = ap.parse_args()

    out = Path(a.out) if a.out else Path(a.weights).parents[1]
    out.mkdir(parents=True, exist_ok=True)
    m = YOLO(a.weights).val(data=a.data, split=a.split, imgsz=a.imgsz, batch=a.batch, device=a.device,
                            conf=a.conf, plots=True, project=str(out), name=f"eval_{a.split}", exist_ok=True)
    res = {"split": a.split, "imgsz": a.imgsz, "weights": a.weights,
           "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
           "precision": float(m.box.mp), "recall": float(m.box.mr),
           "per_class_AP50-95": {m.names[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])}}
    (out / f"metrics_{a.split}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
