"""Fine-tune an Ultralytics YOLO OBB model on the prepared DRASHTI dataset.

Usage (on kaggle-vm, launch detached: nohup python train.py ... > train.log 2>&1 &):
  python train.py --model yolo11m-obb.pt --imgsz 1280 --epochs 40 --batch 8 --device 0,1 --name m1280
  python train.py --name m1280 --resume            # continue an interrupted run
  python train.py --hours 9 ...                    # time budget; overrides --epochs, LR schedule fits the budget
Validation each epoch uses data.yaml's val_subset.txt; run evaluate.py for full val/test.
Outputs: <project>/<name>/weights/{best,last}.pt, results.csv, plots.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/tmp/drashti/data.yaml")
    ap.add_argument("--model", default="yolo11m-obb.pt", help="pretrained OBB weights (or a .pt to continue from)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hours", type=float, default=None, help="time budget in hours (overrides --epochs)")
    ap.add_argument("--batch", type=int, default=8, help="total batch (split across GPUs); -1 = auto")
    ap.add_argument("--device", default="0,1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="/kaggle/working/runs")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--save-period", type=int, default=5, help="also keep epochN.pt every N epochs")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set (for quick probes)")
    ap.add_argument("--flipud", type=float, default=0.5, help="aerial imagery: vertical flip is valid")
    ap.add_argument("--degrees", type=float, default=0.0, help="rotation aug in degrees")
    ap.add_argument("--close-mosaic", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    if a.resume:
        YOLO(str(Path(a.project) / a.name / "weights" / "last.pt")).train(resume=True)
        return
    model = YOLO(a.model)
    kw = dict(data=a.data, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
              project=a.project, name=a.name, patience=a.patience, save_period=a.save_period,
              fraction=a.fraction, flipud=a.flipud, degrees=a.degrees, close_mosaic=a.close_mosaic,
              cos_lr=True, amp=True, plots=True, exist_ok=True)
    if a.hours:
        kw["time"] = a.hours
    else:
        kw["epochs"] = a.epochs
    model.train(**kw)


if __name__ == "__main__":
    main()
